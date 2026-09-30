use bytes::{Bytes, BytesMut};
use serde::Deserialize;
use serde_json::value::RawValue;
use std::collections::HashMap;
use std::ops::Range;
use tracing::warn;

// Bound an unterminated event without truncating upstream bytes.
const MAX_EVENT_BYTES: usize = 16 * 1024 * 1024;

#[derive(Default)]
pub struct ResponsesStream {
    pending: BytesMut,
    scan: usize,
    line_start: usize,
    item_ids: HashMap<u64, String>,
    passthrough: bool,
    warned: bool,
    started: bool,
}

// These are read-only views. Output is assembled from the original byte slices,
// never by serializing a view, so unknown fields and their spelling survive.
#[derive(Deserialize)]
struct EventView<'a> {
    #[serde(rename = "type", borrow)]
    kind: Option<&'a RawValue>,
    #[serde(borrow)]
    output_index: Option<&'a RawValue>,
    #[serde(borrow)]
    item_id: Option<&'a RawValue>,
    #[serde(borrow)]
    item: Option<&'a RawValue>,
    #[serde(borrow)]
    response: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct ItemView<'a> {
    #[serde(borrow)]
    id: Option<&'a RawValue>,
}

#[derive(Deserialize)]
struct ResponseView<'a> {
    #[serde(borrow)]
    output: Option<&'a RawValue>,
}

struct Patch {
    range: Range<usize>,
    replacement: String,
}

impl ResponsesStream {
    pub fn push(&mut self, chunk: Bytes) -> Vec<Bytes> {
        if self.passthrough {
            return vec![chunk];
        }
        self.pending.extend_from_slice(&chunk);
        let mut frames = Vec::new();
        // The scan cursor survives network reads, including a split CRLF.
        while self.scan < self.pending.len() {
            let byte = self.pending[self.scan];
            if byte != b'\r' && byte != b'\n' {
                self.scan += 1;
            } else {
                if byte == b'\r' && self.scan + 1 == self.pending.len() {
                    break;
                }
                let empty = self.scan == self.line_start;
                self.scan += if byte == b'\r' && self.pending.get(self.scan + 1) == Some(&b'\n') {
                    2
                } else {
                    1
                };
                self.line_start = self.scan;
                if empty {
                    let frame = self.pending.split_to(self.scan).freeze();
                    self.scan = 0;
                    self.line_start = 0;
                    frames.push(self.normalize_frame(frame));
                }
            }
            if self.scan > MAX_EVENT_BYTES {
                warn!(
                    "Copilot SSE event exceeds buffer limit; relaying remaining stream unchanged"
                );
                self.passthrough = true;
                self.item_ids.clear();
                frames.push(self.drain());
                break;
            }
        }
        frames
    }

    pub fn finish(&mut self) -> Bytes {
        let frame = self.drain();
        if self.passthrough {
            frame
        } else {
            self.normalize_frame(frame)
        }
    }

    pub fn drain(&mut self) -> Bytes {
        self.scan = 0;
        self.line_start = 0;
        self.pending.split().freeze()
    }

    fn normalize_frame(&mut self, frame: Bytes) -> Bytes {
        let mut data = Vec::new();
        let mut pos = if !self.started && frame.starts_with(b"\xef\xbb\xbf") {
            3
        } else {
            0
        };
        self.started = true;
        while pos < frame.len() {
            let start = pos;
            while pos < frame.len() && frame[pos] != b'\r' && frame[pos] != b'\n' {
                pos += 1;
            }
            let line = &frame[start..pos];
            if line == b"data" {
                data.push(pos..pos);
            } else if line.starts_with(b"data:") {
                let value_start = start + 5 + usize::from(line.get(5) == Some(&b' '));
                data.push(value_start..pos);
            }
            pos += if frame.get(pos) == Some(&b'\r') && frame.get(pos + 1) == Some(&b'\n') {
                2
            } else {
                1
            };
        }
        if data.is_empty() {
            return frame;
        }
        // SSE joins multiple data lines with LF. Retain a map back to the wire
        // so comments, field prefixes and line endings are never regenerated.
        let joined;
        let payload = if data.len() == 1 {
            &frame[data[0].clone()]
        } else {
            joined = data
                .iter()
                .map(|range| &frame[range.clone()])
                .collect::<Vec<_>>()
                .join(&b'\n');
            &joined
        };
        if payload == b"[DONE]" || payload.is_empty() {
            return frame;
        }
        let patches = std::str::from_utf8(payload)
            .ok()
            .and_then(|json| self.patches(json));
        let Some(mut patches) = patches else {
            if !self.warned {
                warn!("Unrecognized Copilot SSE JSON; preserving event bytes");
                self.warned = true;
            }
            return frame;
        };
        if patches.is_empty() {
            return frame;
        }
        patches.sort_by_key(|patch| patch.range.start);
        let mut offset = 0;
        let mut line_index = 0;
        for patch in &mut patches {
            while line_index < data.len() && patch.range.start >= offset + data[line_index].len() {
                offset += data[line_index].len() + 1;
                line_index += 1;
            }
            // A JSON string token cannot contain a literal line break.
            let Some(range) = data.get(line_index) else {
                return frame;
            };
            if patch.range.start < offset || patch.range.end > offset + range.len() {
                return frame;
            }
            patch.range = (range.start + patch.range.start - offset)
                ..(range.start + patch.range.end - offset);
        }
        let mut result = BytesMut::with_capacity(frame.len());
        let mut cursor = 0;
        for patch in patches {
            if patch.range.start < cursor {
                return frame;
            }
            result.extend_from_slice(&frame[cursor..patch.range.start]);
            result.extend_from_slice(patch.replacement.as_bytes());
            cursor = patch.range.end;
        }
        result.extend_from_slice(&frame[cursor..]);
        result.freeze()
    }

    fn patches(&mut self, json: &str) -> Option<Vec<Patch>> {
        let event: EventView<'_> = serde_json::from_str(json).ok()?;
        let kind = event
            .kind
            .and_then(|raw| serde_json::from_str::<String>(raw.get()).ok());
        let mut patches = Vec::new();
        if !kind.is_some_and(|kind| kind.starts_with("response.")) {
            return Some(patches);
        }
        if let Some(index) = event
            .output_index
            .and_then(|raw| serde_json::from_str::<u64>(raw.get()).ok())
        {
            if let Some(item) = event.item {
                self.item_patch(json, index, item, &mut patches);
            }
            if let Some(id) = event.item_id {
                self.id_patch(json, index, id, &mut patches);
            }
        }
        if let Some(response) = event
            .response
            .and_then(|raw| serde_json::from_str::<ResponseView<'_>>(raw.get()).ok())
        {
            if let Some(output) = response
                .output
                .and_then(|raw| serde_json::from_str::<Vec<&RawValue>>(raw.get()).ok())
            {
                for (index, item) in output.into_iter().enumerate() {
                    self.item_patch(json, index as u64, item, &mut patches);
                }
            }
        }
        Some(patches)
    }

    fn item_patch(&mut self, json: &str, index: u64, item: &RawValue, patches: &mut Vec<Patch>) {
        if let Some(id) = serde_json::from_str::<ItemView<'_>>(item.get())
            .ok()
            .and_then(|view| view.id)
        {
            self.id_patch(json, index, id, patches);
        }
    }

    fn id_patch(&mut self, json: &str, index: u64, raw: &RawValue, patches: &mut Vec<Patch>) {
        let Ok(id) = serde_json::from_str::<String>(raw.get()) else {
            return;
        };
        if id.is_empty() {
            return;
        }
        let canonical = self.item_ids.entry(index).or_insert_with(|| id.clone());
        if id == *canonical {
            return;
        }
        // from_str::<&RawValue> borrows the token directly from this JSON input.
        // Integer offsets and checked slicing need no unsafe pointer arithmetic.
        let Some(start) = (raw.get().as_ptr() as usize).checked_sub(json.as_ptr() as usize) else {
            return;
        };
        let Some(end) = start.checked_add(raw.get().len()) else {
            return;
        };
        if json.get(start..end) != Some(raw.get()) {
            return;
        }
        patches.push(Patch {
            range: start..end,
            replacement: serde_json::to_string(canonical).expect("JSON string serialization"),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    fn event(value: Value) -> String {
        format!("data: {value}\n\n")
    }

    fn added(index: u64, id: &str) -> String {
        event(
            json!({"type":"response.output_item.added","output_index":index,"item":{"id":id,"type":"message"}}),
        )
    }

    fn delta(index: u64, id: &str) -> String {
        event(
            json!({"type":"response.output_text.delta","output_index":index,"item_id":id,"delta":"hello"}),
        )
    }

    fn run(source: &[u8], chunk_size: usize) -> Vec<u8> {
        let mut stream = ResponsesStream::default();
        let mut result = Vec::new();
        for chunk in source.chunks(chunk_size) {
            for frame in stream.push(Bytes::copy_from_slice(chunk)) {
                result.extend_from_slice(&frame);
            }
        }
        result.extend_from_slice(&stream.finish());
        result
    }

    #[test]
    fn preserves_every_byte_outside_identity_tokens() {
        let prefix = added(0, "first");
        let raw = concat!(
            "event: response.output_text.delta\r\nid: transport-id\r\n: heartbeat\r\nretry: 20\r\n",
            "data: { \"future\":{\"number\":1.2300e+04,\"huge\":1e999,\"s\":\"\\u4e2d\\/x\"},\r\n",
            "data: \"type\":\"response.output_text.delta\", \"item_\\u0069d\" : \"rotated\",\r\n",
            "data: \"output_index\":0,\"delta\":\"你好🌏 rotated\",\"nested\":{\"item_id\":\"rotated\"} }\r\n\r\n",
        );
        let source = format!("{prefix}{raw}");
        let expected = format!(
            "{prefix}{}",
            raw.replacen("\\u0069d\" : \"rotated\"", "\\u0069d\" : \"first\"", 1)
        );
        for chunk_size in [1, 2, 3, 7, 31, source.len()] {
            assert_eq!(run(source.as_bytes(), chunk_size), expected.as_bytes());
        }
    }

    #[test]
    fn all_chunk_boundaries_and_line_endings_preserve_identity() {
        for newline in ["\n", "\r\n", "\r"] {
            let source = format!("{}{}", added(0, "a"), delta(0, "b")).replace('\n', newline);
            let expected = format!("{}{}", added(0, "a"), delta(0, "a")).replace('\n', newline);
            for split in 0..=source.len() {
                let mut stream = ResponsesStream::default();
                let mut frames = stream.push(Bytes::copy_from_slice(&source.as_bytes()[..split]));
                frames.extend(stream.push(Bytes::copy_from_slice(&source.as_bytes()[split..])));
                frames.push(stream.finish());
                assert_eq!(frames.concat(), expected.as_bytes(), "split={split}");
            }
        }
    }

    #[test]
    fn terminal_snapshots_and_interleaved_outputs_keep_state_and_tool_payloads() {
        for terminal in [
            "response.completed",
            "response.incomplete",
            "response.failed",
        ] {
            let mut source = String::new();
            let mut expected = String::new();
            for index in 0..3 {
                source.push_str(&added(index, &format!("first-{index}")));
                expected.push_str(&added(index, &format!("first-{index}")));
            }
            for index in [2, 0, 1] {
                source.push_str(&delta(index, "rotated"));
                expected.push_str(&delta(index, &format!("first-{index}")));
            }
            let items = vec![
                json!({"type":"reasoning","id":"r-done","encrypted_content":"original-blob","summary":[]}),
                json!({"type":"function_call","id":"t-done","call_id":"original-call","arguments":"{\"id\":\"t-done\"}"}),
                json!({"type":"message","id":"m-done","phase":"commentary","content":[{"text":"same text","type":"output_text"}]}),
            ];
            let mut normalized = items.clone();
            for (index, item) in items.iter().enumerate() {
                source.push_str(&event(
                    json!({"type":"response.output_item.done","output_index":index,"item":item}),
                ));
                normalized[index]["id"] = json!(format!("first-{index}"));
                expected.push_str(&event(json!({"type":"response.output_item.done","output_index":index,"item":normalized[index]})));
            }
            source.push_str(&event(
                json!({"type":terminal,"response":{"id":"response-id","output":items}}),
            ));
            expected.push_str(&event(
                json!({"type":terminal,"response":{"id":"response-id","output":normalized}}),
            ));
            assert_eq!(run(source.as_bytes(), 13), expected.as_bytes());
        }
    }

    #[test]
    fn healthy_unknown_and_malformed_events_pass_through() {
        let source = format!("{}{}{}", added(0, "same"), delta(0, "same"), concat!(
            ": keepalive\n\nevent: ping\n\ndata\n\ndata:\n\ndata: [DONE]\n\n",
            "data: {\n\ndata: []\n\ndata: {\"type\":42,\"item_id\":\"x\"}\n\n",
            "data: {\"type\":\"future.event\",\"output_index\":0,\"item_id\":\"x\"}\n\n",
            "data: {\"type\":\"response.future\",\"output_index\":0,\"item_id\":\"a\",\"item_id\":\"b\"}\n\n",
        ));
        assert_eq!(run(source.as_bytes(), 1), source.as_bytes());
        let invalid_utf8 = b"data: \xff\n\n";
        assert_eq!(run(invalid_utf8, 1), invalid_utf8);
    }

    #[test]
    fn unusable_indices_and_changed_field_shapes_are_not_guessed() {
        for index in [
            json!(-1),
            json!(1.5),
            json!("0"),
            json!(true),
            Value::Null,
            json!({}),
        ] {
            let source = format!(
                "{}{}",
                added(0, "first"),
                event(json!({"type":"response.future","output_index":index,"item_id":"rotated"}))
            );
            assert_eq!(run(source.as_bytes(), 3), source.as_bytes());
        }
        for extra in [
            json!({"item_id":null}),
            json!({"item_id":42}),
            json!({"item_id":""}),
            json!({"item":[]}),
            json!({"item":{"id":false}}),
            json!({"response":{"output":{}}}),
            json!({"response":{"output":[null,42,{}]}}),
        ] {
            let mut value = json!({"type":"response.future","output_index":0});
            value
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let source = format!("{}{}", added(0, "first"), event(value));
            assert_eq!(run(source.as_bytes(), 5), source.as_bytes());
        }
    }

    #[test]
    fn escaped_ids_are_decoded_and_replacements_are_json_encoded() {
        let id = "first\"\\\n你好";
        let source = format!("{}{}", added(0, id), delta(0, "rotated"));
        let expected = format!("{}{}", added(0, id), delta(0, id));
        assert_eq!(run(source.as_bytes(), 2), expected.as_bytes());
        let healthy = format!("{}data: {{\"type\":\"response.future\",\"output_index\":0,\"item_id\":\"\\u0061\"}}\n\n", added(0, "a"));
        assert_eq!(run(healthy.as_bytes(), 7), healthy.as_bytes());
    }

    #[test]
    fn first_reference_snapshot_and_request_isolation_are_supported() {
        let snapshot =
            event(json!({"type":"response.queued","response":{"output":[{"id":"snap"}]}}));
        let created = event(json!({"type":"response.created"}));
        let source = format!(
            "{snapshot}{created}{}{}{}",
            added(0, "later"),
            delta(1, "early"),
            added(1, "later"),
        );
        let expected = format!(
            "{snapshot}{created}{}{}{}",
            added(0, "snap"),
            delta(1, "early"),
            added(1, "early"),
        );
        assert_eq!(run(source.as_bytes(), 11), expected.as_bytes());
        assert_eq!(
            run(added(0, "isolated").as_bytes(), 1),
            added(0, "isolated").as_bytes()
        );
    }

    #[test]
    fn delta_is_forwarded_before_completion_and_eof_does_not_add_bytes() {
        let mut stream = ResponsesStream::default();
        stream.push(Bytes::from(added(0, "first")));
        assert_eq!(
            stream.push(Bytes::from(delta(0, "other"))).concat(),
            delta(0, "first").as_bytes()
        );
        let tail = delta(0, "other").trim_end().to_owned();
        assert!(stream.push(Bytes::from(tail)).is_empty());
        assert_eq!(stream.finish(), delta(0, "first").trim_end().as_bytes());
        assert!(stream.finish().is_empty());
        stream.push(Bytes::from_static(b"data: {\"type\":"));
        assert_eq!(stream.drain(), b"data: {\"type\":".as_slice());
    }

    #[test]
    fn initial_sse_bom_is_preserved_and_does_not_hide_first_item() {
        let source = format!("\u{feff}{}{}", added(0, "first"), delta(0, "other"));
        let expected = format!("\u{feff}{}{}", added(0, "first"), delta(0, "first"));
        assert_eq!(run(source.as_bytes(), 1), expected.as_bytes());
    }

    #[test]
    fn multiple_snapshot_patches_preserve_multiline_sse_offsets() {
        let prefix = format!("{}{}", added(0, "long-first-id"), added(1, "b"));
        let snapshot = json!({"type":"response.completed","response":{"output":[
            {"id":"x","call_id":"x"}, {"id":"much-longer-final-id","encrypted_content":"unchanged"}
        ]}});
        let mut expected = snapshot.clone();
        expected["response"]["output"][0]["id"] = json!("long-first-id");
        expected["response"]["output"][1]["id"] = json!("b");
        let wire = |value: &Value| {
            serde_json::to_string_pretty(value)
                .unwrap()
                .lines()
                .map(|line| format!("data: {line}\r\n: keep\r\n"))
                .collect::<String>()
                + "\r\n"
        };
        assert_eq!(
            run(format!("{prefix}{}", wire(&snapshot)).as_bytes(), 1),
            format!("{prefix}{}", wire(&expected)).as_bytes()
        );
    }

    #[test]
    fn oversized_unterminated_events_switch_to_bounded_passthrough() {
        let oversized = vec![b'x'; MAX_EVENT_BYTES + 1];
        let mut stream = ResponsesStream::default();
        assert_eq!(
            stream.push(Bytes::from(oversized.clone())).concat(),
            oversized
        );
        assert!(stream.pending.is_empty());
        let next = Bytes::from_static(b"data: [DONE]\n\n");
        assert_eq!(stream.push(next.clone()), vec![next]);
        assert!(stream.finish().is_empty());
    }

    #[test]
    #[ignore = "Run with --release --ignored --nocapture for timing"]
    fn benchmark_event_processing() {
        use std::hint::black_box;
        use std::time::Instant;
        for (size, iterations) in [(32, 100_000), (8192, 10_000), (262144, 300)] {
            let canonical = "a".repeat(400);
            let json = json!({"type":"response.output_text.delta","output_index":0,"item_id":"b".repeat(400),"delta":"x".repeat(size)}).to_string();
            let frame = Bytes::from(format!("data: {json}\n\n"));
            let mut stream = ResponsesStream::default();
            stream.push(Bytes::from(added(0, &canonical)));
            let start = Instant::now();
            for _ in 0..iterations {
                black_box(stream.push(black_box(frame.clone())));
            }
            let raw_ns = start.elapsed().as_nanos() / iterations;
            let start = Instant::now();
            for _ in 0..iterations {
                let mut value: Value = serde_json::from_str(black_box(&json)).unwrap();
                value["item_id"] = Value::String(canonical.clone());
                black_box(serde_json::to_vec(&value).unwrap());
            }
            let value_ns = start.elapsed().as_nanos() / iterations;
            println!("bytes={} iterations={iterations} raw_sse_ns={raw_ns} value_json_ns={value_ns} throughput_mib_s={:.1}", frame.len(), frame.len() as f64 * 1e9 / raw_ns as f64 / 1048576.0);
        }
    }
}
