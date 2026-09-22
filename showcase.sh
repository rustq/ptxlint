#!/usr/bin/env bash
# A guided tour of everything ptxlint does, run against the cases in cases/.
#
# CI runs this on a machine with no GPU and no CUDA toolkit, which is the whole
# point of the tool; the log doubles as living documentation. Run it locally
# with `cargo build --release && ./showcase.sh`.
set -u

PTXLINT=${PTXLINT:-./target/release/ptxlint}
cd "$(dirname "$0")"

if [[ ! -x $PTXLINT ]]; then
    echo "no binary at $PTXLINT — run: cargo build --release" >&2
    exit 2
fi

in_ci() { [[ -n ${GITHUB_ACTIONS:-} ]]; }

group() {
    if in_ci; then echo "::group::$*"; else printf '\n\033[1m━━ %s\033[0m\n' "$*"; fi
}
endgroup() { in_ci && echo "::endgroup::"; return 0; }

# Echo the command, run it, and report the exit code — the codes are part of
# what is being demonstrated.
show() {
    printf '\033[2m$ %s\033[0m\n' "$*"
    "$@"
    printf '\033[2m→ exit %d\033[0m\n' "$?"
}

group "1. A kernel with a problem"
show $PTXLINT cases/ptx001_local_memory.ptx
endgroup

group "2. A kernel without one"
show $PTXLINT cases/clean_saxpy.ptx
endgroup

group "3. A whole directory at once"
report=$($PTXLINT cases)
echo "$(ls cases/*.ptx | wc -l) files, $(echo "$report" | grep -cE '^  [a-z]') kernels:"
echo "$report" | grep -E '^  [a-z]|error, '
endgroup

group "4. The architecture changes the verdict"
echo "FP64 is half rate on a datacentre part and 1/64 on GeForce:"
$PTXLINT --arch sm_80 cases/ptx003_fp64_literals.ptx | grep PTX003
$PTXLINT --arch sm_89 cases/ptx003_fp64_literals.ptx | grep PTX003
endgroup

group "5. The block size changes occupancy"
for n in 32 128 256 1024; do
    line=$($PTXLINT --arch sm_86 --block-size $n cases/clean_saxpy.ptx | grep '^    occupancy')
    printf '%5s threads/block -> %s, %s\n' "$n" \
        "$(echo "$line" | grep -o 'occupancy [0-9]*%')" \
        "$(echo "$line" | sed 's/.*limited by //; s/)$//')"
done
endgroup

group "6. Replaying a ptxas log makes the numbers exact"
echo "Without it, registers are an upper bound and spills are invisible:"
$PTXLINT cases/ptx002_register_spills.ptx | grep 'regs/thread'
echo
echo "With a saved 'ptxas -v' log from a machine that has CUDA:"
show $PTXLINT --ptxas-report cases/ptx002_register_spills.ptxas.txt \
    cases/ptx002_register_spills.ptx
endgroup

group "7. Diffing two builds of the same kernel"
echo "The same kernel before and after the local-memory fix:"
show $PTXLINT --baseline cases/diff_before.ptx cases/diff_after.ptx
echo "And the other way round, with the CI gate on:"
show $PTXLINT --deny regression --baseline cases/diff_after.ptx cases/diff_before.ptx
endgroup

group "8. Exit codes for CI"
echo "0 clean, 1 a denied lint fired, 2 ptxlint could not run:"
show $PTXLINT --deny error cases/clean_saxpy.ptx
show $PTXLINT --deny error cases/ptx001_local_memory.ptx
show $PTXLINT --deny PTX003 cases/ptx001_local_memory.ptx
show $PTXLINT no/such/file.ptx
endgroup

group "9. Machine-readable output"
$PTXLINT --json cases/ptx001_local_memory.ptx | head -20
endgroup

group "10. Reading from stdin"
show sh -c "cat cases/ptx004_integer_division.ptx | $PTXLINT - | tail -4"
endgroup

group "11. Instructions it has never seen"
echo "Tensor cores and async copies parse fine without special support:"
$PTXLINT cases/modern_tensor_cores.ptx | sed -n '3,6p'
endgroup

group "12. Every lint"
$PTXLINT --list-lints
endgroup

echo
echo "All 10 lints have a case in cases/; see the README table."
exit 0
