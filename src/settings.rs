use base64::{engine::general_purpose::STANDARD, Engine};
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use ring::rand::{SecureRandom, SystemRandom};
use std::{
    fs::OpenOptions,
    io::{Read, Write},
    path::Path,
};

use crate::config::ProxyConfig;

const AAD: &[u8] = b"claude-proxy-config-v1";
const LIMIT: usize = 1024 * 1024;
const ERROR: &str = "Invalid encrypted proxy configuration";

pub fn write_settings(input: impl Read, path: &Path) -> Result<(), &'static str> {
    let mut plaintext = Vec::new();
    input
        .take(LIMIT as u64 + 1)
        .read_to_end(&mut plaintext)
        .map_err(|_| "Cannot read configuration")?;
    let settings = encrypt(&plaintext)?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .map_err(|_| "Cannot create private settings file")?;
    file.write_all(&settings)
        .map_err(|_| "Cannot write settings file")
}

fn encrypt(plaintext: &[u8]) -> Result<Vec<u8>, &'static str> {
    if plaintext.len() > LIMIT {
        return Err(ERROR);
    }
    let config = parse_plaintext(std::str::from_utf8(plaintext).map_err(|_| ERROR)?)?;
    config.validate_auth()?;
    let random = SystemRandom::new();
    let mut key_bytes = [0; 32];
    let mut nonce_bytes = [0; 12];
    random
        .fill(&mut key_bytes)
        .map_err(|_| "Cannot generate encryption key")?;
    random
        .fill(&mut nonce_bytes)
        .map_err(|_| "Cannot generate encryption nonce")?;
    let key = LessSafeKey::new(UnboundKey::new(&AES_256_GCM, &key_bytes).map_err(|_| ERROR)?);
    let mut ciphertext = plaintext.to_vec();
    key.seal_in_place_append_tag(
        Nonce::assume_unique_for_key(nonce_bytes),
        Aad::from(AAD),
        &mut ciphertext,
    )
    .map_err(|_| ERROR)?;
    let envelope = format!(
        "v1.{}",
        STANDARD.encode([nonce_bytes.to_vec(), ciphertext].concat())
    );
    if envelope.len() > LIMIT {
        return Err(ERROR);
    }
    serde_json::to_vec(&serde_json::json!({
        "CLAUDE_PROXY_CONFIG": envelope,
        "CLAUDE_PROXY_SECRET": STANDARD.encode(key_bytes)
    }))
    .map_err(|_| ERROR)
}

pub fn read(config: &str, secret: Option<&str>) -> Result<ProxyConfig, &'static str> {
    match secret {
        Some(secret) => decrypt(config.trim(), secret),
        None => parse_plaintext(config),
    }
}

fn parse_plaintext(value: &str) -> Result<ProxyConfig, &'static str> {
    let format = if value.trim_start().starts_with('{') {
        config::FileFormat::Json
    } else {
        config::FileFormat::Toml
    };
    let source = config::Config::builder()
        .add_source(config::File::from_str(value, format))
        .add_source(
            config::Environment::default()
                .prefix("CLAUDE_PROXY")
                .separator("__"),
        )
        .build()
        .map_err(|_| "Invalid proxy configuration")?;
    let explicit_upstream = source.get_string("upstream_url").is_ok();
    let mut config: ProxyConfig = source
        .try_deserialize()
        .map_err(|_| "Invalid proxy configuration")?;
    if config.uses_copilot() && !explicit_upstream {
        config.upstream_url.clear();
    }
    Ok(config)
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
    parse_plaintext(std::str::from_utf8(plaintext).map_err(|_| ERROR)?).map_err(|_| ERROR)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAINTEXT: &[u8] = br#"{"upstream_url":"https://api.githubcopilot.com","copilot_pools":{"copilot-1":{"api_key":"fixture-key","github":[{"login":"alice","token":"fixture-token"}]}}}"#;

    #[test]
    fn encryption_roundtrips_with_fresh_random_keys_and_nonces() {
        let first = encrypt(PLAINTEXT).unwrap();
        let second = encrypt(PLAINTEXT).unwrap();
        assert_ne!(first, second);
        assert!(!String::from_utf8_lossy(&first).contains("fixture-token"));
        let settings: serde_json::Value = serde_json::from_slice(&first).unwrap();
        let config = decrypt(
            settings["CLAUDE_PROXY_CONFIG"].as_str().unwrap(),
            settings["CLAUDE_PROXY_SECRET"].as_str().unwrap(),
        )
        .unwrap();
        assert_eq!(
            config.copilot_pools.unwrap().0["copilot-1"].github[0].token,
            "fixture-token"
        );
        for value in [b"invalid-json".as_slice(), b"{}", &vec![b' '; LIMIT + 1]] {
            assert!(encrypt(value).is_err());
        }
        let mut large: serde_json::Value = serde_json::from_slice(PLAINTEXT).unwrap();
        large["padding"] = serde_json::Value::String("x".repeat(800_000));
        assert!(encrypt(&serde_json::to_vec(&large).unwrap()).is_err());
    }

    #[test]
    fn writes_private_file_and_preserves_existing_files() {
        let path =
            std::env::temp_dir().join(format!("proxy-settings-{}.json", uuid::Uuid::new_v4()));
        write_settings(PLAINTEXT, &path).unwrap();
        let original = std::fs::read(&path).unwrap();
        assert!(write_settings(PLAINTEXT, &path).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), original);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        std::fs::remove_file(&path).unwrap();
        assert!(write_settings(b"fixture-secret-invalid".as_slice(), &path).is_err());
        assert!(!path.exists());
    }

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
        let config = read(&envelope, Some(&key)).unwrap();
        assert_eq!(config.port, 8123);
        assert_eq!(
            config.copilot_pools.unwrap().0["copilot-1"].api_key,
            "fixture-key"
        );
        assert!(read(&envelope, None).is_err());
        assert!(read("{}", Some(&key)).is_err());
        assert!(read("{}", None).is_ok());
        assert_eq!(read("port = 8124", None).unwrap().port, 8124);
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
