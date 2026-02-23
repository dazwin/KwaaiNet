//! DID utilities for KwaaiNet
//!
//! KwaaiNet uses `did:peer` for node identities. A node's DID is derived
//! directly from its libp2p PeerId, which is itself derived from an Ed25519
//! keypair. This makes it a **self-certifying identifier** — no external
//! registry is needed to bind a DID to a cryptographic key.
//!
//! ## Format
//! ```text
//! did:peer:<base58-encoded-peer-id>
//! ```
//!
//! ## Key insight
//! The libp2p PeerId is functionally equivalent to a `did:key`. It is already
//! used as the Layer 1 identity anchor throughout the KwaaiNet DHT. This module
//! provides the glue between the libp2p world and the W3C DID world.

use libp2p::PeerId;

/// Convert a libp2p `PeerId` to a `did:peer:` DID string
///
/// # Example
/// ```ignore
/// // did:peer:QmYyQSo1c1Ym7orWxLYvCuxRjeczyuq4GNGbMaFfkMhp4
/// let did = peer_id_to_did(&peer_id);
/// ```
pub fn peer_id_to_did(peer_id: &PeerId) -> String {
    format!("did:peer:{}", peer_id.to_base58())
}

/// Extract a `PeerId` from a `did:peer:` DID string
///
/// Returns `None` if the DID is not in `did:peer:` format or the base58 is invalid.
pub fn did_to_peer_id(did: &str) -> Option<PeerId> {
    did.strip_prefix("did:peer:")
        .and_then(|base58| base58.parse().ok())
}

/// Returns `true` if the given DID string corresponds to the given `PeerId`
pub fn did_matches_peer(did: &str, peer_id: &PeerId) -> bool {
    did_to_peer_id(did)
        .map(|p| p == *peer_id)
        .unwrap_or(false)
}

/// Construct the W3C verification method URI for a node's primary key
///
/// Format: `did:peer:<base58>#key-1`
///
/// This is used in the `verificationMethod` field of a `CredentialProof`.
pub fn verification_method(peer_id: &PeerId) -> String {
    format!("{}#key-1", peer_id_to_did(peer_id))
}

/// Extract the raw 32-byte Ed25519 public key from a libp2p PeerId
///
/// A libp2p PeerId for an Ed25519 key is a multihash of the protobuf-encoded
/// public key:
/// ```text
/// identity_multihash( protobuf{ key_type=Ed25519, data=<32 bytes> } )
/// ```
///
/// The protobuf pattern is: `\x08\x01\x12\x20` + 32 key bytes.
/// The multihash wrapper prepends `\x00` (identity code) + varint(length).
///
/// Returns `None` for PeerIds that do not encode an Ed25519 public key
/// (e.g., SHA256-hashed RSA keys that predate the identity-multihash scheme).
pub fn extract_ed25519_bytes(peer_id: &PeerId) -> Option<[u8; 32]> {
    let bytes = peer_id.to_bytes();
    // Scan for the protobuf field header:
    //   field 1 (key_type), wire type 0, value 1 (Ed25519) → 0x08 0x01
    //   field 2 (data),     wire type 2, length 32         → 0x12 0x20
    for i in 0..bytes.len().saturating_sub(35) {
        if bytes[i] == 0x08
            && bytes[i + 1] == 0x01
            && bytes[i + 2] == 0x12
            && bytes[i + 3] == 0x20
        {
            return bytes[i + 4..i + 36].try_into().ok();
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_peer_id_did() {
        // Generate a fresh keypair and PeerId
        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let peer_id = keypair.public().to_peer_id();

        let did = peer_id_to_did(&peer_id);
        assert!(did.starts_with("did:peer:"));

        let recovered = did_to_peer_id(&did).expect("should round-trip");
        assert_eq!(recovered, peer_id);
    }

    #[test]
    fn did_matches_peer_positive() {
        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let peer_id = keypair.public().to_peer_id();
        let did = peer_id_to_did(&peer_id);
        assert!(did_matches_peer(&did, &peer_id));
    }

    #[test]
    fn extract_ed25519_roundtrip() {
        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let peer_id = keypair.public().to_peer_id();
        let key_bytes = extract_ed25519_bytes(&peer_id).expect("should extract Ed25519 bytes");
        assert_eq!(key_bytes.len(), 32);
    }
}
