# License

Trueno-DB is released under the **MIT License**, one of the most permissive open-source licenses.

## What This Means for You

The MIT License grants you the freedom to:

- ✅ **Use commercially** - Deploy Trueno-DB in commercial products without fees
- ✅ **Modify freely** - Adapt the code to your specific needs
- ✅ **Distribute** - Share original or modified versions
- ✅ **Sublicense** - Include in proprietary software
- ✅ **Private use** - Use internally without publishing changes

The only requirements:

- 📄 **Include the license** - Keep copyright notice in distributions
- ⚖️ **No warranty** - Software provided "as is" without guarantees

## Full License Text

```text
MIT License

Copyright (c) 2025 Pragmatic AI Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Why MIT License?

We chose the MIT License for Trueno-DB to:

1. **Encourage adoption** - Zero barriers to commercial or academic use
2. **Foster innovation** - No copyleft restrictions on derivative works
3. **Industry standard** - Compatible with most corporate legal policies
4. **Simple compliance** - Just include the license text

## Dependency Licenses

Trueno-DB builds on excellent open-source libraries. Key dependencies and their licenses:

### Core Dependencies

| Dependency | Version | License | Purpose |
|------------|---------|---------|---------|
| `trueno` | 0.6.0 | MIT | SIMD compute library (AVX-512/AVX2) |
| `wgpu` | 22 | MIT/Apache-2.0 | GPU compute (WebGPU backend) |
| `arrow` | 53 | Apache-2.0 | Columnar data format |
| `parquet` | 53 | Apache-2.0 | Parquet file reader |
| `sqlparser` | 0.52 | Apache-2.0 | SQL query parsing |
| `tokio` | 1 | MIT | Async runtime |
| `rayon` | 1.8 | MIT/Apache-2.0 | CPU parallelism |

### License Compatibility

All dependencies use permissive licenses (MIT, Apache-2.0) that are fully compatible with commercial use:

- **MIT License**: Allows unrestricted use with attribution
- **Apache-2.0**: Similar to MIT with explicit patent grant

**Result**: You can use Trueno-DB in commercial products without copyleft concerns.

## Third-Party Notices

When distributing Trueno-DB, include license notices for:

1. **Apache Arrow** - Apache License 2.0 (columnar format)
2. **wgpu** - Dual-licensed MIT/Apache-2.0 (GPU backend)
3. **trueno** - MIT License (SIMD library)
4. **Other dependencies** - See `Cargo.toml` for complete list

Run this command to generate a full dependency license report:

```bash
cargo install cargo-license
cargo license --authors --do-not-bundle
```

## Attribution

If you use Trueno-DB in your project, we appreciate (but don't require) attribution:

> Powered by [Trueno-DB](https://github.com/paiml/trueno-db) - GPU-first embedded analytics

## Contributing

By contributing to Trueno-DB, you agree that your contributions will be licensed under the same MIT License. See [Contributing Guide](../dev/contributing.md) for details.

## Questions?

For licensing questions, contact:
- **Email**: info@paiml.com
- **GitHub Issues**: [trueno-db/issues](https://github.com/paiml/trueno-db/issues)

## Additional Resources

- [OSI - MIT License](https://opensource.org/licenses/MIT) - Official MIT License page
- [TLDRLegal - MIT](https://www.tldrlegal.com/license/mit-license) - Plain-English summary
- [Choose a License](https://choosealicense.com/licenses/mit/) - MIT License guide
- [SPDX - MIT](https://spdx.org/licenses/MIT.html) - SPDX identifier reference

## See Also

- **[Contributing Guide](../dev/contributing.md)** - How to contribute code
- **[Code of Conduct](../dev/contributing.md#code-of-conduct)** - Community guidelines
- **[Security Policy](https://github.com/paiml/trueno-db/security)** - Reporting vulnerabilities
