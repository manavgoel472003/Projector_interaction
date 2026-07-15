.PHONY: install model test previews run clean

install:
	./install.sh

model:
	./scripts/download_hand_model.sh

test:
	.venv/bin/python -m unittest discover -s tests -v

previews:
	.venv/bin/python scripts/render_mode_previews.py

run:
	./run_wall_touch_demo.sh

clean:
	rm -rf build dist *.egg-info __pycache__ tests/__pycache__
