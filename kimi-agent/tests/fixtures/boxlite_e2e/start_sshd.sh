#!/bin/sh
set -eu

exec /usr/sbin/sshd -D -f /root/fixtures/sshd_config -E /tmp/sshd.log
