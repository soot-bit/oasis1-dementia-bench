.PHONY: prep bench-tab bench-cnn-best

prep:
	bash scripts/prep_oasis1.sh

bench-tab:
	bash scripts/bench_tab.sh

bench-cnn-best:
	bash scripts/bench_cnn_best.sh
