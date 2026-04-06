# References

## Foundational Papers (Methodology)

1. Popper, K. (1959). *The Logic of Scientific Discovery*. Hutchinson & Co.
2. Shingo, S. (1986). *Zero Quality Control: Source Inspection and the Poka-Yoke System*. Productivity Press.
3. Meyer, B. (1988). *Object-Oriented Software Construction*. Prentice Hall.
4. Brady, E. (2017). *Type-Driven Development with Idris*. Manning Publications.
5. King, A. (2019). "Parse, Don't Validate." <https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/>
6. Williams, S., Waterman, A., Patterson, D. (2009).
   "Roofline: An Insightful Visual Performance Model." CACM, 52(4).
7. Wulf, W. & McKee, S. (1995). "Hitting the Memory Wall: Implications of the Obvious." ACM SIGARCH, 23(1).

## ML Kernel Papers (Target Contracts)

8. Vaswani, A. et al. (2017). "Attention Is All You Need." arXiv:1706.03762.
9. Zhang, B. & Sennrich, R. (2019). "Root Mean Square Layer Normalization." arXiv:1910.10683.
10. Su, J. et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864.
11. Shazeer, N. (2020). "GLU Variants Improve Transformer." arXiv:2002.05202.
12. Dao, T. et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact Attention." arXiv:2205.14135.
13. Dao, T. (2023). "FlashAttention-2: Faster Attention with Better Parallelism."
    arXiv:2307.08691.
14. Pope, R. et al. (2022). "Efficiently Scaling Transformer Inference." arXiv:2210.09461.
15. Frantar, E. et al. (2022). "GPTQ: Accurate Post-Training Quantization." arXiv:2210.17323.
16. Dettmers, T. et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." NeurIPS 2022.
17. Holtzman, A. et al. (2019). "The Curious Case of Neural Text Degeneration." arXiv:1904.09751.
18. Milakov, M. & Gimelshein, N. (2018). "Online Normalizer Calculation for Softmax." arXiv:1805.02867.
19. Rabe, M. & Staats, C. (2021). "Self-Attention Does Not Need O(n^2) Memory." arXiv:2112.05682.

## Formal Verification

20. Kani Contributors (2022-2025). "Kani Rust Verifier." <https://github.com/model-checking/kani>
21. VanHattum, A. et al. (2022). "Verifying Dynamic Trait Objects in Rust." ICSE-SEIP 2022.
22. Chong, N. et al. (2021). "Code-Level Model Checking in the Software
    Development Workflow." ICSE-SEIP 2021. (CBMC foundations)
23. AWS Security (2023). "Using Kani to Validate Security Boundaries
    in AWS Firecracker." model-checking.github.io/kani-verifier-blog/
24. AWS (2023). "How s2n-quic Uses Kani to Inspire Confidence."
    model-checking.github.io/kani-verifier-blog/
25. Rust Standard Library Verification (2025). "Verifying the Rust Standard Library." arXiv:2510.01072.

## PAIML Stack Components

26. **trueno** -- SIMD-accelerated tensor operations. <https://crates.io/crates/trueno>
27. **aprender** -- Machine learning library. <https://crates.io/crates/aprender>
28. **realizar** -- Inference engine. <https://crates.io/crates/realizar>
29. **certeza** -- Quality validation framework.
30. **probar** -- Property-based testing with metamorphic relations.
31. **batuta** -- Workflow orchestrator.
32. **pmat** -- Code quality toolkit.
