#!/usr/bin/env python3
"""
Filter labour rules by context variables.

Separates rules into:
1. Rules with at least 1 SUPPORTED context variable
2. Rules with only UNSUPPORTED or CUSTOM (llm) variables

Generates statistics about the split.
"""

import json
import os
from datetime import datetime
from collections import defaultdict

# Define SUPPORTED variables (from client specification)
SUPPORTED_VARIABLES = {
    'schedule_type',
    'day_of_week',
    'daily_hours',
    'weekly_hours',
    'daily_overtime_hours',
    'shift_consecutive_days',
    'is_foreman',
    'is_stat_holiday',
    'project_type',
    'shift_type',
    'employee_class',
    'position_classification',
    'validity_start_date',
    'validity_end_date'
}

# Define UNSUPPORTED variables (from client specification)
UNSUPPORTED_VARIABLES = {
    'week_number',
    'project_location',
    'employee_status',
    'claim_type',
    'is_local_resident',
    'project_distance_from_hiring_hall',
    'project_distance_from_city_hall_km',
    'job_location_radius_km',
    'jurisdiction',
    'event',
    'work_commenced',
    'work_start_status',
    'activity_type',
    'is_absence_preauthorized',
    'is_irregular_single_shift',
    'is_makeup_day',
    'union_local',
    'test_result',
    'test'
}


def parse_context_variable(var_string):
    """
    Parse a context variable string to extract variable name and type.

    Returns: (variable_name, variable_type)
    variable_type: 'supported', 'unsupported', or 'custom'
    """
    var_string = var_string.strip()

    # Check for unsupported marker
    if '(Unsupported)' in var_string:
        var_name = var_string.replace('(Unsupported)', '').strip()
        return var_name, 'unsupported'

    # Check for LLM/custom marker
    if '(llm)' in var_string:
        var_name = var_string.replace('(llm)', '').strip()
        return var_name, 'custom'

    # No marker - check if it's in supported list
    if var_string in SUPPORTED_VARIABLES:
        return var_string, 'supported'

    # No marker but not in supported list - treat as custom
    return var_string, 'custom'


def classify_rule(rule):
    """
    Classify a rule based on its context variables.

    Returns: (has_supported, variable_details)
    """
    context_vars = rule.get('context_variables', [])

    supported_vars = []
    unsupported_vars = []
    custom_vars = []

    for var in context_vars:
        var_name, var_type = parse_context_variable(var)

        if var_type == 'supported':
            supported_vars.append(var_name)
        elif var_type == 'unsupported':
            unsupported_vars.append(var_name)
        else:
            custom_vars.append(var_name)

    has_supported = len(supported_vars) > 0

    return has_supported, {
        'supported': supported_vars,
        'unsupported': unsupported_vars,
        'custom': custom_vars
    }


def generate_statistics(rules, variable_details_list):
    """Generate detailed statistics about the rules."""
    stats = {
        'total_rules': len(rules),
        'variable_usage': defaultdict(int),
        'supported_variable_counts': defaultdict(int),
        'unsupported_variable_counts': defaultdict(int),
        'custom_variable_counts': defaultdict(int),
        'rules_by_topic': defaultdict(int),
        'rules_with_no_context_vars': 0
    }

    for i, rule in enumerate(rules):
        # Count by topic
        topic = rule.get('primary_topic', 'UNKNOWN')
        stats['rules_by_topic'][topic] += 1

        # Check for empty context vars
        if not rule.get('context_variables'):
            stats['rules_with_no_context_vars'] += 1

        # Count variable usage
        details = variable_details_list[i]
        for var in details['supported']:
            stats['supported_variable_counts'][var] += 1
            stats['variable_usage'][var] += 1
        for var in details['unsupported']:
            stats['unsupported_variable_counts'][var] += 1
            stats['variable_usage'][var] += 1
        for var in details['custom']:
            stats['custom_variable_counts'][var] += 1
            stats['variable_usage'][var] += 1

    # Convert defaultdicts to regular dicts for JSON serialization
    stats['variable_usage'] = dict(sorted(stats['variable_usage'].items(), key=lambda x: -x[1]))
    stats['supported_variable_counts'] = dict(sorted(stats['supported_variable_counts'].items(), key=lambda x: -x[1]))
    stats['unsupported_variable_counts'] = dict(sorted(stats['unsupported_variable_counts'].items(), key=lambda x: -x[1]))
    stats['custom_variable_counts'] = dict(sorted(stats['custom_variable_counts'].items(), key=lambda x: -x[1]))
    stats['rules_by_topic'] = dict(sorted(stats['rules_by_topic'].items(), key=lambda x: -x[1]))

    return stats


def main():
    # File paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'labour_rules_extracted.json')

    # Output files
    supported_output = os.path.join(script_dir, 'rules_with_supported_variables.json')
    unsupported_custom_output = os.path.join(script_dir, 'rules_with_unsupported_custom_variables.json')
    stats_output = os.path.join(script_dir, 'filter_statistics.json')

    print("=" * 70)
    print("RULE FILTERING BY CONTEXT VARIABLES")
    print("=" * 70)
    print(f"\nInput file: {input_file}")

    # Load rules
    with open(input_file, 'r', encoding='utf-8') as f:
        all_rules = json.load(f)

    print(f"Total rules loaded: {len(all_rules)}")

    # Classify rules
    rules_with_supported = []
    rules_without_supported = []

    supported_var_details = []
    unsupported_var_details = []

    for rule in all_rules:
        has_supported, var_details = classify_rule(rule)

        if has_supported:
            rules_with_supported.append(rule)
            supported_var_details.append(var_details)
        else:
            rules_without_supported.append(rule)
            unsupported_var_details.append(var_details)

    # Generate statistics
    overall_stats = {
        'generated_at': datetime.now().isoformat(),
        'input_file': os.path.basename(input_file),
        'total_rules': len(all_rules),
        'summary': {
            'rules_with_supported_variables': len(rules_with_supported),
            'rules_without_supported_variables': len(rules_without_supported),
            'percentage_with_supported': round(len(rules_with_supported) / len(all_rules) * 100, 2) if all_rules else 0,
            'percentage_without_supported': round(len(rules_without_supported) / len(all_rules) * 100, 2) if all_rules else 0
        },
        'supported_rules_stats': generate_statistics(rules_with_supported, supported_var_details),
        'unsupported_custom_rules_stats': generate_statistics(rules_without_supported, unsupported_var_details),
        'supported_variables_reference': sorted(list(SUPPORTED_VARIABLES)),
        'unsupported_variables_reference': sorted(list(UNSUPPORTED_VARIABLES))
    }

    # Save files
    print(f"\n{'=' * 70}")
    print("SAVING OUTPUT FILES")
    print("=" * 70)

    with open(supported_output, 'w', encoding='utf-8') as f:
        json.dump(rules_with_supported, f, indent=2, ensure_ascii=False)
    print(f"\n[1] Rules with SUPPORTED variables: {supported_output}")
    print(f"    Count: {len(rules_with_supported)} rules")

    with open(unsupported_custom_output, 'w', encoding='utf-8') as f:
        json.dump(rules_without_supported, f, indent=2, ensure_ascii=False)
    print(f"\n[2] Rules with UNSUPPORTED/CUSTOM variables only: {unsupported_custom_output}")
    print(f"    Count: {len(rules_without_supported)} rules")

    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, indent=2, ensure_ascii=False)
    print(f"\n[3] Statistics: {stats_output}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nTotal Rules: {len(all_rules)}")
    print(f"\n┌─────────────────────────────────────────────────────────────────────┐")
    print(f"│ Rules WITH supported variables:    {len(rules_with_supported):>5} ({overall_stats['summary']['percentage_with_supported']:>6.2f}%)           │")
    print(f"│ Rules WITHOUT supported variables: {len(rules_without_supported):>5} ({overall_stats['summary']['percentage_without_supported']:>6.2f}%)           │")
    print(f"└─────────────────────────────────────────────────────────────────────┘")

    # Top supported variables used
    if overall_stats['supported_rules_stats']['supported_variable_counts']:
        print(f"\n{'─' * 70}")
        print("TOP SUPPORTED VARIABLES IN USE:")
        print("─" * 70)
        for var, count in list(overall_stats['supported_rules_stats']['supported_variable_counts'].items())[:10]:
            print(f"  • {var}: {count} rules")

    # Top unsupported variables
    all_unsupported = {}
    for var, count in overall_stats['supported_rules_stats']['unsupported_variable_counts'].items():
        all_unsupported[var] = all_unsupported.get(var, 0) + count
    for var, count in overall_stats['unsupported_custom_rules_stats']['unsupported_variable_counts'].items():
        all_unsupported[var] = all_unsupported.get(var, 0) + count

    if all_unsupported:
        print(f"\n{'─' * 70}")
        print("TOP UNSUPPORTED VARIABLES IN USE:")
        print("─" * 70)
        for var, count in sorted(all_unsupported.items(), key=lambda x: -x[1])[:10]:
            print(f"  • {var}: {count} rules")

    # Top custom variables
    all_custom = {}
    for var, count in overall_stats['supported_rules_stats']['custom_variable_counts'].items():
        all_custom[var] = all_custom.get(var, 0) + count
    for var, count in overall_stats['unsupported_custom_rules_stats']['custom_variable_counts'].items():
        all_custom[var] = all_custom.get(var, 0) + count

    if all_custom:
        print(f"\n{'─' * 70}")
        print("TOP CUSTOM (LLM) VARIABLES IN USE:")
        print("─" * 70)
        for var, count in sorted(all_custom.items(), key=lambda x: -x[1])[:10]:
            print(f"  • {var}: {count} rules")

    # Rules by topic breakdown
    print(f"\n{'─' * 70}")
    print("RULES BY PRIMARY TOPIC:")
    print("─" * 70)
    print(f"\n{'Topic':<40} {'Supported':<12} {'Unsupported':<12}")
    print(f"{'-'*40} {'-'*12} {'-'*12}")

    all_topics = set(overall_stats['supported_rules_stats']['rules_by_topic'].keys()) | \
                 set(overall_stats['unsupported_custom_rules_stats']['rules_by_topic'].keys())

    for topic in sorted(all_topics):
        supported_count = overall_stats['supported_rules_stats']['rules_by_topic'].get(topic, 0)
        unsupported_count = overall_stats['unsupported_custom_rules_stats']['rules_by_topic'].get(topic, 0)
        print(f"{topic:<40} {supported_count:<12} {unsupported_count:<12}")

    print(f"\n{'=' * 70}")
    print("FILTERING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
