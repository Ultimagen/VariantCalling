REPO_ROOT=$(dirname $0)
if [ -e ${REPO_ROOT}/.git/lfs/ ]; then
  if grep -r git-lfs ${REPO_ROOT}/test/resources; then
    echo "Git-lfs is installed, but you need to run 'git-lfs pull' to pull large files from remote"
  else
    python -m pytest ${REPO_ROOT}/test
  fi
else
  echo "Missing git lfs installation, please install git lfs https://git-lfs.github.com/ before running tests"
fi
