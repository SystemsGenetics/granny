
build:
	docker buildx build -t systemsgenetics/granny .

push:
	docker push systemsgenetics/granny . 
