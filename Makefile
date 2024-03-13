default:


install:
	@pip install -r requirements.txt

run_api:
	@uvicorn api.api:app --reload
