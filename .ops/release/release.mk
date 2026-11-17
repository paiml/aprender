# release.mk - include from a repo's Makefile. REPO, SHA, REL_DIR, REL_PINS set by caller.
REL_LIB ?= $(dir $(lastword $(MAKEFILE_LIST)))release-lib.sh
REL_PINS ?= $(dir $(lastword $(MAKEFILE_LIST)))pins.tsv
release-gate:
	@. $(REL_LIB) && rel_gate "$(REPO)" "$(SHA)" "$(REL_DIR)" "$(REL_PINS)"
.PHONY: release-gate
