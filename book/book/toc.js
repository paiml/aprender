// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="introduction.html">Introduction</a></li><li class="chapter-item expanded affix "><li class="part-title">Core Concepts</li><li class="chapter-item expanded "><a href="methodology/what-is-pacha.html"><strong aria-hidden="true">1.</strong> What is Pacha?</a></li><li class="chapter-item expanded "><a href="methodology/mlops-artifacts.html"><strong aria-hidden="true">2.</strong> MLOps Artifact Management</a></li><li class="chapter-item expanded "><a href="methodology/semantic-versioning.html"><strong aria-hidden="true">3.</strong> Semantic Versioning for ML</a></li><li class="chapter-item expanded "><a href="methodology/reproducibility.html"><strong aria-hidden="true">4.</strong> Reproducibility Principles</a></li><li class="chapter-item expanded affix "><li class="part-title">Architecture</li><li class="chapter-item expanded "><a href="architecture/overview.html"><strong aria-hidden="true">5.</strong> System Overview</a></li><li class="chapter-item expanded "><a href="architecture/content-addressed-storage.html"><strong aria-hidden="true">6.</strong> Content-Addressed Storage</a></li><li class="chapter-item expanded "><a href="architecture/sqlite-metadata.html"><strong aria-hidden="true">7.</strong> SQLite Metadata Store</a></li><li class="chapter-item expanded "><a href="architecture/blake3-hashing.html"><strong aria-hidden="true">8.</strong> BLAKE3 Hashing</a></li><li class="chapter-item expanded affix "><li class="part-title">Model Registry</li><li class="chapter-item expanded "><a href="registry/model-versioning.html"><strong aria-hidden="true">9.</strong> Model Versioning</a></li><li class="chapter-item expanded "><a href="registry/model-cards.html"><strong aria-hidden="true">10.</strong> Model Cards</a></li><li class="chapter-item expanded "><a href="registry/lifecycle-stages.html"><strong aria-hidden="true">11.</strong> Lifecycle Stages</a></li><li class="chapter-item expanded "><a href="registry/model-lineage.html"><strong aria-hidden="true">12.</strong> Model Lineage</a></li><li class="chapter-item expanded affix "><li class="part-title">Data Registry</li><li class="chapter-item expanded "><a href="registry/dataset-versioning.html"><strong aria-hidden="true">13.</strong> Dataset Versioning</a></li><li class="chapter-item expanded "><a href="registry/datasheets.html"><strong aria-hidden="true">14.</strong> Datasheets</a></li><li class="chapter-item expanded "><a href="registry/data-provenance.html"><strong aria-hidden="true">15.</strong> Data Provenance</a></li><li class="chapter-item expanded affix "><li class="part-title">Recipe Registry</li><li class="chapter-item expanded "><a href="registry/training-recipes.html"><strong aria-hidden="true">16.</strong> Training Recipes</a></li><li class="chapter-item expanded "><a href="registry/hyperparameters.html"><strong aria-hidden="true">17.</strong> Hyperparameters</a></li><li class="chapter-item expanded "><a href="registry/environment-deps.html"><strong aria-hidden="true">18.</strong> Environment Dependencies</a></li><li class="chapter-item expanded affix "><li class="part-title">Storage Layer</li><li class="chapter-item expanded "><a href="storage/content-addressing.html"><strong aria-hidden="true">19.</strong> Content Addressing</a></li><li class="chapter-item expanded "><a href="storage/deduplication.html"><strong aria-hidden="true">20.</strong> Deduplication</a></li><li class="chapter-item expanded "><a href="storage/integrity-verification.html"><strong aria-hidden="true">21.</strong> Integrity Verification</a></li><li class="chapter-item expanded "><a href="storage/compression.html"><strong aria-hidden="true">22.</strong> Compression</a></li><li class="chapter-item expanded affix "><li class="part-title">Lineage Tracking</li><li class="chapter-item expanded "><a href="lineage/lineage-graph.html"><strong aria-hidden="true">23.</strong> Lineage Graph</a></li><li class="chapter-item expanded "><a href="lineage/model-derivation.html"><strong aria-hidden="true">24.</strong> Model Derivation</a></li><li class="chapter-item expanded "><a href="lineage/fine-tuning.html"><strong aria-hidden="true">25.</strong> Fine-Tuning Lineage</a></li><li class="chapter-item expanded "><a href="lineage/quantization.html"><strong aria-hidden="true">26.</strong> Quantization Tracking</a></li><li class="chapter-item expanded affix "><li class="part-title">CLI Reference</li><li class="chapter-item expanded "><a href="cli/installation.html"><strong aria-hidden="true">27.</strong> Installation</a></li><li class="chapter-item expanded "><a href="cli/model-commands.html"><strong aria-hidden="true">28.</strong> Model Commands</a></li><li class="chapter-item expanded "><a href="cli/data-commands.html"><strong aria-hidden="true">29.</strong> Data Commands</a></li><li class="chapter-item expanded "><a href="cli/recipe-commands.html"><strong aria-hidden="true">30.</strong> Recipe Commands</a></li><li class="chapter-item expanded "><a href="cli/run-commands.html"><strong aria-hidden="true">31.</strong> Run Commands</a></li><li class="chapter-item expanded affix "><li class="part-title">Examples</li><li class="chapter-item expanded "><a href="examples/quick-start.html"><strong aria-hidden="true">32.</strong> Quick Start</a></li><li class="chapter-item expanded "><a href="examples/registering-models.html"><strong aria-hidden="true">33.</strong> Registering Models</a></li><li class="chapter-item expanded "><a href="examples/tracking-experiments.html"><strong aria-hidden="true">34.</strong> Tracking Experiments</a></li><li class="chapter-item expanded "><a href="examples/managing-datasets.html"><strong aria-hidden="true">35.</strong> Managing Datasets</a></li><li class="chapter-item expanded "><a href="examples/training-workflows.html"><strong aria-hidden="true">36.</strong> Training Workflows</a></li><li class="chapter-item expanded affix "><li class="part-title">Best Practices</li><li class="chapter-item expanded "><a href="best-practices/versioning-strategy.html"><strong aria-hidden="true">37.</strong> Versioning Strategy</a></li><li class="chapter-item expanded "><a href="best-practices/model-documentation.html"><strong aria-hidden="true">38.</strong> Model Documentation</a></li><li class="chapter-item expanded "><a href="best-practices/reproducibility-checklist.html"><strong aria-hidden="true">39.</strong> Reproducibility Checklist</a></li><li class="chapter-item expanded "><a href="best-practices/cicd-integration.html"><strong aria-hidden="true">40.</strong> CI/CD Integration</a></li><li class="chapter-item expanded affix "><li class="part-title">Appendix</li><li class="chapter-item expanded "><a href="appendix/glossary.html"><strong aria-hidden="true">41.</strong> Glossary</a></li><li class="chapter-item expanded "><a href="appendix/references.html"><strong aria-hidden="true">42.</strong> References</a></li><li class="chapter-item expanded "><a href="appendix/api-reference.html"><strong aria-hidden="true">43.</strong> API Reference</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString();
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
