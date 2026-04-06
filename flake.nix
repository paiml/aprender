{
  description = "Realizar - Pure Rust ML Inference Engine";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };

        # Pin Rust version for reproducibility
        rustVersion = pkgs.rust-bin.stable."1.83.0".default.override {
          extensions = [ "rust-src" "rust-analyzer" "llvm-tools" ];
          targets = [ "wasm32-unknown-unknown" ];
        };

        # Build dependencies
        buildInputs = with pkgs; [
          openssl
          pkg-config
        ] ++ lib.optionals stdenv.isDarwin [
          darwin.apple_sdk.frameworks.Security
          darwin.apple_sdk.frameworks.SystemConfiguration
        ];

        # Development tools
        devTools = with pkgs; [
          rustVersion
          cargo-watch
          cargo-edit
          cargo-outdated
          cargo-audit
          cargo-deny
          cargo-llvm-cov
          cargo-mutants
          cargo-criterion

          # Python for benchmarks
          python311
          python311Packages.torch
          python311Packages.numpy

          # Documentation
          mdbook

          # Utilities
          jq
          wrk
          hyperfine
        ];
      in
      {
        # Development shell
        devShells.default = pkgs.mkShell {
          buildInputs = buildInputs ++ devTools;

          shellHook = ''
            echo "Realizar development environment"
            echo "Rust: $(rustc --version)"
            echo "Cargo: $(cargo --version)"

            # Set git commit template
            git config commit.template .gitmessage 2>/dev/null || true

            # Enable CPU performance governor if available
            if command -v cpupower &> /dev/null; then
              echo "Note: Run 'sudo cpupower frequency-set --governor performance' for consistent benchmarks"
            fi
          '';

          RUST_BACKTRACE = "1";
          RUST_LOG = "info";
        };

        # Package
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "realizar";
          version = "0.2.3";

          src = ./.;

          cargoLock = {
            lockFile = ./Cargo.lock;
          };

          nativeBuildInputs = with pkgs; [ pkg-config ];
          buildInputs = buildInputs;

          # Skip tests in Nix build (run separately)
          doCheck = false;

          meta = with pkgs.lib; {
            description = "Pure Rust ML Inference Engine";
            homepage = "https://github.com/paiml/realizar";
            license = licenses.mit;
            maintainers = [ ];
          };
        };

        # Docker image
        packages.docker = pkgs.dockerTools.buildImage {
          name = "realizar";
          tag = "latest";

          copyToRoot = pkgs.buildEnv {
            name = "image-root";
            paths = [ self.packages.${system}.default ];
            pathsToLink = [ "/bin" ];
          };

          config = {
            Cmd = [ "/bin/realizar" "serve" "--demo" "--host" "0.0.0.0" "--port" "3000" ];
            ExposedPorts = {
              "3000/tcp" = {};
            };
            Env = [
              "RUST_LOG=info"
            ];
          };
        };
      }
    );
}
