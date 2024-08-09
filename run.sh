#!/bin/bash

DETACH="--detach"
EXPORT_DB=0
PROFILER_PARAMS=""
NO_CACHE=""
SCRIPT=""
SCRIPT2=""
SERVICES_TO_BUILD="cookie_crawler"
SERVICE=""
BUILD_ALL=0
NOT_FOUND=false
DOCKER_COMPOSE_CMD="run"
COMPOSE_CMD=""
STOP_CONTAINERS=0

# Check for docker compose command
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "Neither 'docker-compose' nor 'docker compose' is installed."
    echo "Please install Docker and Docker Compose to proceed."
    echo 'Make sure to add your user to the docker group (`sudo usermod -a -G docker $USER`).'
    exit 1
fi

cd docker/

while [[ $NOT_FOUND = false ]]
do
    case "$1" in
        --test)
	        DOCKER_COMPOSE_CMD=""
            SCRIPT="python cookie_crawler/run_crawler.py"
            SERVICE="cookie_crawler"
            shift
            ;;

        --crawler)
            SCRIPT="python cookie_crawler/run_crawler.py"
            SERVICE="cookie_crawler"
            shift
            ;;

        --predictor)
            SCRIPT="python classifiers/predict_cookies.py"
            SCRIPT2="python classifiers/predict_purposes.py"
            SERVICE="classifiers"
            shift
            ;;

        --summary)
            SCRIPT="python cookie_crawler/crawl_summary.py"
            SERVICE="cookie_crawler"
            DETACH=""
            shift
            ;;

        --down)
            STOP_CONTAINERS=1
            shift
            ;;

         --ml-eval)
            SERVICE="ml-eval"
            SCRIPT="bash ./classifiers/ml_eval.sh"
            DETACH=""
            shift
            ;;

        --ls)
            SCRIPT="python database/queries.py --command ls"
            SERVICE="cookie_crawler"
            DETACH=""
            shift
            ;;

        --fg)
            DETACH=""
            shift
            ;;

        --no_cache)
            NO_CACHE="--no-cache"
            shift
            ;;

        --script)
            SCRIPT=$2
            shift 2
            ;;

        --build_all)
            BUILD_ALL=1
            SERVICES_TO_BUILD=""
            shift
            ;;

        *)
            NOT_FOUND=true
            ;;

    esac
done

if [ $STOP_CONTAINERS == 1 ]; then
    $COMPOSE_CMD down
    exit 0
fi

if [ -z $SERVICE ]; then
    echo "Please use one of the --crawler, --predictor, --summary, or --down flags."
    exit 1
fi

if [ $BUILD_ALL == 0 ]; then
    SERVICES_TO_BUILD=$SERVICE
fi

$COMPOSE_CMD build $NO_CACHE $SERVICES_TO_BUILD

if [[ -n $DOCKER_COMPOSE_CMD ]]; then
    $COMPOSE_CMD $DOCKER_COMPOSE_CMD $DETACH $SERVICE $SCRIPT $*

    if [[ -n $SCRIPT2 ]]; then
        docker wait $(docker ps -l | cut -d' ' -f1 | tail -n 1)
        $COMPOSE_CMD $DOCKER_COMPOSE_CMD $DETACH $SERVICE $SCRIPT2 $*
    fi
fi
