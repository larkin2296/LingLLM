freeze:
	pip freeze > requirements.txt

train:
	python train_main.py

eval:
	python eval_with_gpt.py

augment:
	python data_aug_chatgpt.py

loop:
	python run_auto_loop.py

plot:
	python plot_log.py

clean:
	rm -rf logs/*.txt logs/*.csv weights/*.pth