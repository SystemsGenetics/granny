VERSION=dev

build:
	docker build -t systemsgenetics/granny:$(VERSION) .

push:
	docker push systemsgenetics/granny:$(VERSION)