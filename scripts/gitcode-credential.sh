#!/bin/sh
# GitCode 的 git 凭据助手。
#
# 从 ~/.gitcode_token 读取访问令牌，因此令牌不会写进 .git/config 或 remote URL。
# 令牌过期时只需覆盖那个文件，git 配置不用动：
#     printf '%s' '新令牌' > ~/.gitcode_token
#
# 安装（每台机器一次，在本仓库根目录执行）：
#     git config --global gitcode.username "你的GitCode用户名"
#     git config --global credential.https://gitcode.com.helper ""
#     git config --global --add credential.https://gitcode.com.helper "!$PWD/scripts/gitcode-credential.sh"
#
# 中间那条空值不能省：git 的凭据助手是多层叠加的，系统级默认有
# Git Credential Manager（manager），它会抢在前面返回一份 GitCode 不认的凭据，
# push 会报 "HTTP Basic: Access denied"。空值把继承来的列表清空，后面的 --add 才生效。
#
# git 的调用约定：第一个参数是 get / store / erase。
# 只实现 get（提供凭据）；store / erase 直接返回成功，
# 这样 git 不会再把凭据另存到 ~/.git-credentials 之类的地方。

case "$1" in
  get)
    echo "username=$(git config --global gitcode.username)"
    echo "password=$(cat "$HOME/.gitcode_token")"
    ;;
  *)
    exit 0
    ;;
esac
