use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metadados associados a uma entrada no cache para particionamento e auditoria.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Metadata {
    pub tenant_id: String,
    pub model: String,
    pub tags: Vec<String>,
    pub attributes: HashMap<String, String>,
}

impl Metadata {
    pub fn new(tenant_id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            model: model.into(),
            tags: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn with_attribute(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), val.into());
        self
    }

    pub fn matches_tenant(&self, expected_tenant: &str) -> bool {
        self.tenant_id.is_empty() || self.tenant_id == expected_tenant
    }
}
