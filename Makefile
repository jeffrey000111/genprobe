.PHONY: test smoke-fake clean-fake

# Unit tests: sharding, resume, metric interface, config validation. No GPU needed.
test:
	python -m unittest discover -s tests -t . -v

# End-to-end dry run on CPU with fake images: exercises sharding, resume and manifests
# for all four model configs. Safe on a login node. Output goes to data/_fake/.
smoke-fake:
	@for c in configs/gen/*.yaml; do \
	  for k in 0 1; do \
	    python -m genprobe.generate --config $$c --prompts data/prompts/smoke.jsonl \
	      --backend fake --seeds 0 --shard $$k --num-shards 2 || exit 1; \
	  done; \
	done
	@echo "fake images: $$(find data/_fake/images -name '*.png' | wc -l) (expected 80)"

clean-fake:
	rm -rf data/_fake
