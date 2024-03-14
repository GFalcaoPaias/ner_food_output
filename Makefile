default:


install:
	@pip install -r requirements.txt

run_api:
	@uvicorn api.api:app --reload

cloud:
	@docker build -t $GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/$DOCKER_REPO_NAME/$DOCKER_IMAGE_NAME:0.1 .
