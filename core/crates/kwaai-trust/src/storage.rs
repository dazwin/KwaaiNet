//! Credential store at `~/.kwaainet/credentials/`
//!
//! Credentials are stored as individual JSON files — one per VC.
//! File names encode the credential type and issuance date so they sort
//! chronologically and are human-readable.
//!
//! The store is intentionally simple: no database, no index, just files.
//! VCs are small (< 2 KB each) and nodes rarely hold more than a dozen.

use crate::credential::VerifiableCredential;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

/// Persistent store for Verifiable Credentials
pub struct CredentialStore {
    dir: PathBuf,
}

impl CredentialStore {
    /// Create a store rooted at `dir` (directory is created if absent)
    pub fn new(dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating credentials directory: {}", dir.display()))?;
        Ok(Self { dir })
    }

    /// Default store location: `~/.kwaainet/credentials/`
    pub fn default_dir() -> PathBuf {
        kwaainet_home().join("credentials")
    }

    /// Open the store at the default location
    pub fn open_default() -> Result<Self> {
        Self::new(Self::default_dir())
    }

    /// Save a VC to the store (overwrites if the same filename already exists)
    pub fn save(&self, vc: &VerifiableCredential) -> Result<()> {
        let filename = vc_filename(vc);
        let path = self.dir.join(&filename);
        let json = serde_json::to_string_pretty(vc)?;
        std::fs::write(&path, json)
            .with_context(|| format!("writing credential: {}", path.display()))?;
        debug!("Saved credential: {}", filename);
        Ok(())
    }

    /// Import a VC from a JSON file path, validate structure, and store it
    pub fn import_file(&self, path: &Path) -> Result<VerifiableCredential> {
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("reading credential file: {}", path.display()))?;
        let vc: VerifiableCredential =
            serde_json::from_str(&json).context("parsing credential JSON")?;
        self.save(&vc)?;
        Ok(vc)
    }

    /// Load all VCs in the store (malformed files are skipped with a warning)
    pub fn load_all(&self) -> Vec<VerifiableCredential> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        let mut vcs = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            match std::fs::read_to_string(&path) {
                Ok(json) => match serde_json::from_str::<VerifiableCredential>(&json) {
                    Ok(vc) => vcs.push(vc),
                    Err(e) => warn!("Skipping malformed credential {}: {}", path.display(), e),
                },
                Err(e) => warn!("Could not read {}: {}", path.display(), e),
            }
        }
        vcs
    }

    /// Load all VCs whose `credentialSubject.id` matches `subject_did`
    pub fn load_for_subject(&self, subject_did: &str) -> Vec<VerifiableCredential> {
        self.load_all()
            .into_iter()
            .filter(|vc| vc.subject_did() == subject_did)
            .collect()
    }

    /// Load non-expired VCs for a subject DID
    pub fn load_valid_for_subject(&self, subject_did: &str) -> Vec<VerifiableCredential> {
        self.load_for_subject(subject_did)
            .into_iter()
            .filter(|vc| !vc.is_expired())
            .collect()
    }

    /// Path to the credentials directory
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// Derive a deterministic filename for a VC.
///
/// Format: `<type>-<YYYYMMDD-HHMMSS>-<issuer-prefix>.json`
///
/// This gives files that sort by type+issuance and are identifiable at a glance.
fn vc_filename(vc: &VerifiableCredential) -> String {
    let vc_type = vc
        .kwaai_type()
        .map(|t| t.as_str().to_lowercase())
        .unwrap_or_else(|| "credential".to_string());

    let issued = vc.issuance_date.format("%Y%m%d-%H%M%S");

    // Short issuer prefix for disambiguation
    let issuer_prefix = vc
        .issuer_did()
        .strip_prefix("did:peer:")
        .map(|b| b.chars().take(8).collect::<String>())
        .unwrap_or_else(|| "unknown".to_string());

    format!("{}-{}-{}.json", vc_type, issued, issuer_prefix)
}

/// `~/.kwaainet/`
fn kwaainet_home() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".kwaainet")
}
