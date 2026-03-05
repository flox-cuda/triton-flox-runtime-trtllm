#!/usr/bin/env bats
# tests/lib.bats — Unit tests for _lib.sh functions.

setup() {
  source "$(dirname "$BATS_TEST_FILENAME")/test_helper.bash"
  common_setup
  # Unset the guard so we can re-source _lib.sh
  unset _LIB_SH_LOADED
  source "$SCRIPTS_DIR/_lib.sh"
}

teardown() {
  common_teardown
}

# --- lib::is_truthy ---

@test "is_truthy: true returns 0" {
  lib::is_truthy "true"
}

@test "is_truthy: 1 returns 0" {
  lib::is_truthy "1"
}

@test "is_truthy: yes returns 0" {
  lib::is_truthy "yes"
}

@test "is_truthy: TRUE returns 0" {
  lib::is_truthy "TRUE"
}

@test "is_truthy: false returns 1" {
  run lib::is_truthy "false"
  [ "$status" -eq 1 ]
}

@test "is_truthy: 0 returns 1" {
  run lib::is_truthy "0"
  [ "$status" -eq 1 ]
}

@test "is_truthy: no returns 1" {
  run lib::is_truthy "no"
  [ "$status" -eq 1 ]
}

@test "is_truthy: empty returns 1" {
  run lib::is_truthy ""
  [ "$status" -eq 1 ]
}

@test "is_truthy: banana returns 1" {
  run lib::is_truthy "banana"
  [ "$status" -eq 1 ]
}

# --- lib::require_bool ---

@test "require_bool: accepts true" {
  TESTVAR=true
  lib::require_bool TESTVAR
}

@test "require_bool: accepts false" {
  TESTVAR=false
  lib::require_bool TESTVAR
}

@test "require_bool: accepts 1" {
  TESTVAR=1
  lib::require_bool TESTVAR
}

@test "require_bool: accepts 0" {
  TESTVAR=0
  lib::require_bool TESTVAR
}

@test "require_bool: accepts yes" {
  TESTVAR=yes
  lib::require_bool TESTVAR
}

@test "require_bool: accepts no" {
  TESTVAR=no
  lib::require_bool TESTVAR
}

@test "require_bool: rejects banana" {
  TESTVAR=banana
  run lib::require_bool TESTVAR
  [ "$status" -ne 0 ]
}

@test "require_bool: rejects empty" {
  TESTVAR=""
  run lib::require_bool TESTVAR
  [ "$status" -ne 0 ]
}

# --- lib::require_pos_int ---

@test "require_pos_int: accepts 1" {
  TESTVAR=1
  lib::require_pos_int TESTVAR
}

@test "require_pos_int: accepts 9000" {
  TESTVAR=9000
  lib::require_pos_int TESTVAR
}

@test "require_pos_int: rejects 0" {
  TESTVAR=0
  run lib::require_pos_int TESTVAR
  [ "$status" -ne 0 ]
}

@test "require_pos_int: rejects negative" {
  TESTVAR=-1
  run lib::require_pos_int TESTVAR
  [ "$status" -ne 0 ]
}

@test "require_pos_int: rejects non-integer" {
  TESTVAR=abc
  run lib::require_pos_int TESTVAR
  [ "$status" -ne 0 ]
}

@test "require_pos_int: rejects empty" {
  TESTVAR=""
  run lib::require_pos_int TESTVAR
  [ "$status" -ne 0 ]
}

# --- lib::slugify ---

@test "slugify: slashes become hyphens" {
  result="$(lib::slugify "org/model")"
  [ "$result" = "org-model" ]
}

@test "slugify: preserves dots and underscores" {
  result="$(lib::slugify "my_model.v2")"
  [ "$result" = "my_model.v2" ]
}

@test "slugify: collapses special chars" {
  result="$(lib::slugify "a@#\$b")"
  [ "$result" = "a-b" ]
}
