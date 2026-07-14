.PHONY: install model test run clean

install:
	./install.sh

model:
	./scripts/download_hand_model.sh

test:
	.venv/bin/python -m unittest discover -s tests -v

run:
	./run_wall_touch_demo.sh

clean:
	rm -rf build dist *.egg-info __pycache__ tests/__pycache__
