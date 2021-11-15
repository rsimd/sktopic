# from: https://shunyaueta.com/posts/2021-05-23/#poetry-%E3%81%AB%E3%82%88%E3%82%8B%E5%AE%9F%E7%8F%BE%E6%96%B9%E6%B3%95
# package name
PACKAGE = sktopic
.PHONY: build-package

build-package: ## Generate setup.py by poetry command for shared package 
	poetry build
	# source ./dist で解凍
	tar zxvf dist/$(PACKAGE)-*.tar.gz -C ./dist
	# setup.py を手元にコピー
	cp dist/$(PACKAGE)-*/setup.py setup.py
	# poetry build で生成されたファイルをすべて削除
	#rm -rf dist