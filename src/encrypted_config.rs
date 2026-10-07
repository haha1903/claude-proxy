use base64::{engine::general_purpose::STANDARD, Engine};
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};

use crate::config::ProxyConfig;

const AAD: &[u8] = b"claude-proxy-config-v1";
const LIMIT: usize = 1024 * 1024;
const ERROR: &str = "Invalid encrypted proxy configuration";

pub fn from_env() -> Result<Option<ProxyConfig>, &'static str> {
    let read = |name| match std::env::var(name) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(_) => Err(ERROR),
    };
    load(
        read("CLAUDE_PROXY_ENCRYPTED_CONFIG")?.as_deref(),
        read("CLAUDE_PROXY_CONFIG_KEY")?.as_deref(),
    )
}

fn load(envelope: Option<&str>, key: Option<&str>) -> Result<Option<ProxyConfig>, &'static str> {
    match (envelope, key) {
        (None, None) => Ok(None),
        (Some(envelope), Some(key)) => decrypt(envelope, key).map(Some),
        _ => Err(ERROR),
    }
}

fn decrypt(envelope: &str, key: &str) -> Result<ProxyConfig, &'static str> {
    if envelope.len() > LIMIT || key.len() != 44 {
        return Err(ERROR);
    }
    let key = STANDARD.decode(key).map_err(|_| ERROR)?;
    let key = LessSafeKey::new(UnboundKey::new(&AES_256_GCM, &key).map_err(|_| ERROR)?);
    let mut bytes = STANDARD
        .decode(envelope.strip_prefix("v1.").ok_or(ERROR)?)
        .map_err(|_| ERROR)?;
    if bytes.len() < 12 + AES_256_GCM.tag_len() {
        return Err(ERROR);
    }
    let (nonce, ciphertext) = bytes.split_at_mut(12);
    let nonce = Nonce::try_assume_unique_for_key(nonce).map_err(|_| ERROR)?;
    let plaintext = key
        .open_in_place(nonce, Aad::from(AAD), ciphertext)
        .map_err(|_| ERROR)?;
    // Never include parsing errors: serde errors can contain decrypted credentials.
    serde_json::from_slice(plaintext).map_err(|_| ERROR)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seal(plaintext: &[u8], aad: &[u8]) -> (String, String) {
        let key_bytes = [7; 32];
        let nonce = [3; 12];
        let key = LessSafeKey::new(UnboundKey::new(&AES_256_GCM, &key_bytes).unwrap());
        let mut ciphertext = plaintext.to_vec();
        key.seal_in_place_append_tag(
            Nonce::assume_unique_for_key(nonce),
            Aad::from(aad),
            &mut ciphertext,
        )
        .unwrap();
        (
            format!(
                "v1.{}",
                STANDARD.encode([nonce.to_vec(), ciphertext].concat())
            ),
            STANDARD.encode(key_bytes),
        )
    }

    #[test]
    fn loads_authenticated_config_and_requires_both_settings() {
        let (envelope, key) = seal(br#"{"port":8123,"copilot_pools":{"copilot-1":{"api_key":"fixture-key","github":[{"login":"alice","token":"fixture-token"}]}}}"#, AAD);
        let config = load(Some(&envelope), Some(&key)).unwrap().unwrap();
        assert_eq!(config.port, 8123);
        assert_eq!(
            config.copilot_pools.unwrap().0["copilot-1"].api_key,
            "fixture-key"
        );
        assert!(load(None, None).unwrap().is_none());
        assert!(load(Some(&envelope), None).is_err());
        assert!(load(None, Some(&key)).is_err());
    }

    #[test]
    fn rejects_wrong_keys_tampering_versions_and_invalid_plaintext() {
        let (envelope, key) = seal(b"{}", AAD);
        assert!(decrypt(&envelope, &STANDARD.encode([8; 32])).is_err());
        for bad in ["", "v2.AAAA", "v1.!", "v1.AAAA", &"x".repeat(LIMIT + 1)] {
            assert!(decrypt(bad, &key).is_err());
        }
        for bad in ["", &"!".repeat(44), &STANDARD.encode([0; 31])] {
            assert!(decrypt(&envelope, bad).is_err());
        }
        let mut bytes = STANDARD.decode(&envelope[3..]).unwrap();
        for index in [0, 12, bytes.len() - 1] {
            bytes[index] ^= 1;
            assert!(decrypt(&format!("v1.{}", STANDARD.encode(&bytes)), &key).is_err());
            bytes[index] ^= 1;
        }
        let (wrong_aad, _) = seal(b"{}", b"other-app");
        assert!(decrypt(&wrong_aad, &key).is_err());
        let (invalid, _) = seal(br#"{"port":"secret-value"}"#, AAD);
        assert_eq!(decrypt(&invalid, &key).unwrap_err(), ERROR);
    }
}
