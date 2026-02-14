    #[cfg(test)]
    pub(crate) fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }
