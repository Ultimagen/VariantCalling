#!/bin/bash

DOCKER_BASE_NAME="us-central1-docker.pkg.dev/ultima-data-307918/ultimagen/ugvc:"

function usage () {
  echo -e "
  ./build_vc_image.sh -v <git_tag_or_hash> -t <docker_tag> (-h | --help)

  -v | --version    git tag or hash to checkout VariantCalling
  -t | --tag        tag with the version number for resulting docker (e.g. 1.1)
  -base | --docker_base_name docker base name (default is : \"${DOCKER_BASE_NAME}\" )
  -h | --help       print usage and exit
" >&2
  exit $1
}

function parse_args () {
  silent_output=false
	extra_args=""

	while [ "$#" -gt 0 ]; do
	  case "$1" in
	    -v | --version )
			if [ -z $2 ] || [ $(echo $2 | head -c 1) == "-" ]; then
	    		echo -e "\033[1m$1 requires git branch/tag/commit to checkout\033[0m" >&2
	    		usage 1
	    	else
	    		version=$2
    	fi; shift 2;;

	    -t | --tag )
			if [ -z $2 ] || [ $(echo $2 | head -c 1) == "-" ]; then
	    		echo -e "\033[1m$1 requires docker tag\033[0m" >&2
	    		usage 1
	    	else
	    		tag=$2
    	fi; shift 2;;

      -base | --docker_base_name )
      if [ -z $2 ] || [ $(echo $2 | head -c 1) == "-" ]; then
	    		echo -e "\033[1m$1 requires container registry path\033[0m" >&2
	    		usage 1
	    	else
	    		docker_base_name=$2
    	fi; shift 2;;

      -h | --help ) usage 0;;

	    *) echo -e "\033[1mUnknown option or argument: $1, run $0 -h |--help for usage\033[0m" >&2; exit 1;;
	  esac
	done
}

parse_args "$@"

if [ -z "${version}" ]; then
  read -p "Git version not specified, do you want to build docker with current commit? ($(git log -1 --oneline))? [y/n] " confirm
  if [ "${confirm}" != "y" ] && [ "${confirm}" != "Y" ]; then
    echo -e "\033[1mAbort build\033[0m" >&2
    exit 1
  fi
fi

if [ -z "${tag}" ]; then
  read -p "Git tag not specified, specify one: " tag_line
  if [ "${tag_line}" == "" ]; then
    echo -e "\033[1mTag cannot be empty. Abort build\033[0m" >&2
    exit 1
  else
    tag="${tag_line}"
  fi
fi

if [ -z "${docker_base_name}" ]; then
  docker_base_name="${DOCKER_BASE_NAME}"
  echo -e "\033[1mdocker base name not specified, using default: $docker_base_name\033[0m" >&2
fi

abs_script_path=$(realpath $0 | xargs -I {} dirname {} )

if [ -z "${version}" ]; then
  version_for_docker="$(git log -1 --oneline | awk '{ print $1 }')"
else
  cd "${abs_script_path}"
  echo -e "\033[1mMemorize current branch/commit\033[0m"
  current_branch=$(git rev-parse --abbrev-ref HEAD)
  current_commit=$(git log -1 --oneline | awk '{ print $1 }')
  echo -e "\033[1mCheckout to master\033[0m"
  git checkout master
  echo -e "\033[1mPull and fetch all tags\033[0m"
  git pull
  pull_rc=$?

  if [ "${pull_rc}" -gt 0 ]; then
    echo -e "\033[1mGit pull failed. Abort build\033[0m" >&2
    exit 1
  else
    echo -e "\033[1mCheckout to ${version}\033[0m"
    git checkout ${version}
  fi
fi

echo -e "\033[1mMemorize commit hash\033[0m"
commit_hash=$(git log -1 --oneline | awk '{ print $1 }')
version_for_docker="${commit_hash:0:6}"

docker_name="${docker_base_name}${tag}_${version_for_docker}"
echo -e "\033[1mComplete docker name will be: ${docker_name}\033[0m"
docker build . -f Dockerfile.jbvc -t ${docker_name}

if [ -n "${version}" ]; then
  if [ "${current_branch}" == "HEAD" ]; then
    start_state=${current_commit}
  else
    start_state=${current_branch}
  fi
  echo -e "\033[1mGo back to starting state ${start_state}\033[0m"
  git checkout ${start_state}
fi
