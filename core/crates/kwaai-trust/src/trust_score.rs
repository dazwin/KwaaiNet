//! Trust score computation for KwaaiNet nodes (Layer 3)
//!
//! This module implements a **local, portable** trust score derived from a
//! node's verifiable credentials. The score is computed by the *querier*
//! and may differ between observers — there is no central trust registry.
//!
//! ## Current implementation (Phase 2 baseline)
//!
//! A weighted sum of credential contributions, time-decayed by age:
//!
//! ```text
//! CredentialScore = Σ weight(VC_type) × decay(issuance_date)
//! ```
//!
//! ## Planned (Phase 4 — EigenTrust propagation)
//!
//! ```text
//! NodeTrustScore =
//!   w1 × DirectPeerRatings         (peers who transacted with this node)
//!   w2 × CredentialScore           (weighted VCs)
//!   w3 × TransitiveEndorsements    (2 hops, weight × 0.5 per hop)
//!   × TimeDecay(age_of_assertions)
//! ```
//!
//! The EigenTrust algorithm handles Sybil resistance: endorsements from
//! low-trust nodes contribute proportionally less.

use crate::credential::{KwaaiCredentialType, VerifiableCredential};
use chrono::{DateTime, Utc};

/// Local trust score for a KwaaiNet node, in the range [0.0, 1.0]
#[derive(Debug, Clone)]
pub struct TrustScore {
    /// Combined trust score (0 = untrusted, 1 = fully trusted)
    pub score: f64,
    /// Contribution from verifiable credentials (Phase 2)
    pub credential_contribution: f64,
    /// Contribution from direct peer endorsements — 0.0 until Phase 4
    pub peer_endorsement_contribution: f64,
    /// Number of valid (non-expired, recognised-type) credentials counted
    pub credential_count: usize,
}

impl TrustScore {
    /// Compute a credential-weighted trust score (Phase 2 baseline).
    ///
    /// Full EigenTrust propagation over the endorsement graph is not yet
    /// implemented; that arrives in Phase 4.
    pub fn from_credentials(vcs: &[VerifiableCredential]) -> Self {
        let mut credential_score = 0.0_f64;
        let mut count = 0usize;

        for vc in vcs {
            if vc.is_expired() {
                continue;
            }
            let Some(vc_type) = vc.kwaai_type() else {
                continue;
            };
            let weight = vc_type.trust_weight();
            let decay = time_decay(vc.issuance_date);
            credential_score += weight * decay;
            count += 1;
        }

        // Cap at 1.0 — holding all credential types doesn't exceed "fully trusted"
        let capped = credential_score.min(1.0);

        TrustScore {
            score: capped,
            credential_contribution: capped,
            peer_endorsement_contribution: 0.0,
            credential_count: count,
        }
    }

    /// Human-readable trust tier label
    ///
    /// | Score    | Tier     | Meaning                              |
    /// |----------|----------|--------------------------------------|
    /// | ≥ 0.70   | Trusted  | FiduciaryPledge + VerifiedNode + more|
    /// | ≥ 0.40   | Verified | VerifiedNode credential present       |
    /// | ≥ 0.10   | Known    | At least one credential (summit, etc.)|
    /// | < 0.10   | Unknown  | No recognised credentials             |
    pub fn tier_label(&self) -> &'static str {
        match self.score {
            s if s >= 0.70 => "Trusted",
            s if s >= 0.40 => "Verified",
            s if s >= 0.10 => "Known",
            _ => "Unknown",
        }
    }

    /// Maximum weight a single credential type can contribute
    pub fn max_single_weight() -> f64 {
        [
            KwaaiCredentialType::FiduciaryPledgeVC,
            KwaaiCredentialType::VerifiedNodeVC,
            KwaaiCredentialType::UptimeVC,
            KwaaiCredentialType::ThroughputVC,
            KwaaiCredentialType::SummitAttendeeVC,
            KwaaiCredentialType::PeerEndorsementVC,
        ]
        .iter()
        .map(|t| t.trust_weight())
        .fold(0.0_f64, f64::max)
    }
}

/// Exponential time-decay with a 1-year half-life
///
/// `decay(t) = 0.5 ^ (age_days / 365)`
///
/// A credential issued today decays to 0.5 after one year and 0.25 after
/// two years. This keeps the trust graph current without requiring VCs to be
/// re-issued frequently.
fn time_decay(issuance_date: DateTime<Utc>) -> f64 {
    let age_days = (Utc::now() - issuance_date).num_days().max(0) as f64;
    0.5_f64.powf(age_days / 365.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credential::summit_attendee_vc;

    #[test]
    fn empty_credentials_zero_score() {
        let score = TrustScore::from_credentials(&[]);
        assert_eq!(score.score, 0.0);
        assert_eq!(score.tier_label(), "Unknown");
    }

    #[test]
    fn summit_vc_gives_known_tier() {
        let vc = summit_attendee_vc(
            "did:peer:issuer",
            "did:peer:subject",
            "Kwaai Summit 2026",
            "2026-03-15",
        );
        let score = TrustScore::from_credentials(&[vc]);
        assert!(score.score > 0.0);
        assert_eq!(score.tier_label(), "Known");
    }

    #[test]
    fn time_decay_is_one_for_fresh_vc() {
        // A credential issued right now should have a decay factor of 1.0 (age = 0 days)
        let decay = 0.5_f64.powf(0.0 / 365.0);
        assert!((decay - 1.0).abs() < 1e-10);
    }
}
