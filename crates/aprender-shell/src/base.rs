impl CommandGenerator {

    /// Generate synthetic commands
    pub fn generate(&self, count: usize) -> Vec<String> {
        let mut commands = Vec::with_capacity(count);
        let mut seen = HashSet::new();

        // Phase 1-4: progressively more specific commands
        let phases: Vec<Box<dyn Iterator<Item = String> + '_>> = vec![
            Box::new(self.base_commands()),
            Box::new(self.variant_commands()),
            Box::new(self.flag_commands()),
            Box::new(self.full_commands()),
        ];

        for phase in phases {
            for cmd in phase {
                if commands.len() >= count {
                    break;
                }
                if seen.insert(cmd.clone()) {
                    commands.push(cmd);
                }
            }
        }

        commands.truncate(count);
        commands
    }

    /// Phase 1: Base commands from all templates (ensures diversity)
    fn base_commands(&self) -> impl Iterator<Item = String> + '_ {
        self.templates.iter().map(|t| t.base.to_string())
    }

    /// Phase 2: Base + variant from all templates
    fn variant_commands(&self) -> impl Iterator<Item = String> + '_ {
        self.templates.iter().flat_map(|t| {
            t.variants
                .iter()
                .map(move |v| Self::build_command(t.base, v, "", ""))
        })
    }

    /// Phase 3: Base + variant + flag from all templates
    fn flag_commands(&self) -> impl Iterator<Item = String> + '_ {
        self.templates.iter().flat_map(|t| {
            t.variants.iter().flat_map(move |v| {
                t.flags
                    .iter()
                    .map(move |f| Self::build_command(t.base, v, f, ""))
            })
        })
    }

    /// Phase 4: Base + variant + flag + arg (most expansive)
    fn full_commands(&self) -> impl Iterator<Item = String> + '_ {
        self.templates.iter().flat_map(|t| {
            t.variants.iter().flat_map(move |v| {
                t.flags.iter().flat_map(move |f| {
                    t.args
                        .iter()
                        .map(move |a| Self::build_command(t.base, v, f, a))
                })
            })
        })
    }

    /// Build a command string from parts, skipping empty components.
    fn build_command(base: &str, variant: &str, flag: &str, arg: &str) -> String {
        let mut cmd = base.to_string();
        for part in [variant, flag, arg] {
            if !part.is_empty() {
                cmd.push(' ');
                cmd.push_str(part);
            }
        }
        cmd
    }
}
