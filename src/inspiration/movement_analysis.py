"""
Movement Analysis Module

Analyzes employee movement patterns between regions in the bakery.
Identifies frequent transitions, compares against floor connections,
and extracts employee-specific movement patterns.
"""

from collections import defaultdict, Counter

def analyze_movements(data_dict):
    """
    Analyze employee movement paths between regions
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing processed data and metadata
    
    Returns:
    --------
    dict
        Dictionary containing movement analysis results
    """
    data = data_dict['sorted_data']
    connections = data_dict['connections']
    name_to_uuid = data_dict['name_to_uuid']
    uuid_to_name = data_dict['uuid_to_name']
    
    print("  Analyzing transitions between regions...")
    
    # Create employee -> date -> transitions mapping
    employee_transitions = defaultdict(lambda: defaultdict(list))
    all_transitions = []
    
    # Track transitions by employee and date
    for (emp_id, date), group in data.groupby(['id', 'date']):
        group_sorted = group.sort_values('startTime')
        prev_region = None
        prev_time = None
        date_str = date.strftime('%Y-%m-%d')
        
        for _, row in group_sorted.iterrows():
            if prev_region and prev_region != row['region']:
                transition = (prev_region, row['region'])
                transition_time = row['startTime'] - prev_time if prev_time else 0
                
                # Store the transition
                employee_transitions[emp_id][date_str].append({
                    'from': prev_region,
                    'to': row['region'],
                    'time': row['startTime'],
                    'duration': transition_time
                })
                
                all_transitions.append(transition)
            
            prev_region = row['region']
            prev_time = row['startTime']
    
    # Count transitions
    transition_counts = Counter(all_transitions)
    
    print(f"  Identified {len(transition_counts)} unique transitions")
    
    # Check transitions against floor connections
    valid_transitions = {}
    invalid_transitions = {}
    
    for (from_region, to_region), count in transition_counts.items():
        # Convert region names to UUIDs
        from_uuid = name_to_uuid.get(from_region)
        to_uuid = name_to_uuid.get(to_region)
        
        if from_uuid and to_uuid:
            # Check if this is a valid connection in the floor plan
            is_valid = to_uuid in connections.get(from_uuid, [])
            
            if is_valid:
                valid_transitions[(from_region, to_region)] = count
            else:
                invalid_transitions[(from_region, to_region)] = count
    
    print(f"  Valid transitions: {len(valid_transitions)}")
    print(f"  Invalid transitions: {len(invalid_transitions)}")
    
    # Get top transitions
    top_transitions = sorted(valid_transitions.items(), key=lambda x: x[1], reverse=True)
    top_invalid = sorted(invalid_transitions.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate employee-specific patterns
    employee_patterns = {}
    for emp_id in employee_transitions:
        # Get all transitions for this employee
        emp_transitions = []
        for date_transitions in employee_transitions[emp_id].values():
            for t in date_transitions:
                emp_transitions.append((t['from'], t['to']))
        
        # Count and get top transitions
        emp_counts = Counter(emp_transitions)
        top_emp_transitions = sorted(emp_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        employee_patterns[emp_id] = top_emp_transitions
    
    # Calculate region connectivity (for identifying hubs)
    region_connectivity = {}
    for region in data['region'].unique():
        outgoing = sum(count for (from_r, to_r), count in transition_counts.items() if from_r == region)
        incoming = sum(count for (from_r, to_r), count in transition_counts.items() if to_r == region)
        region_connectivity[region] = {
            'outgoing': outgoing,
            'incoming': incoming,
            'total': outgoing + incoming
        }
    
    # Identify hub regions (high connectivity)
    hub_regions = sorted(region_connectivity.items(), key=lambda x: x[1]['total'], reverse=True)
    
    # Identify common movement sequences (patterns of 3+ regions)
    movement_sequences = defaultdict(int)
    seq_length = 3  # Look for sequences of 3 regions
    
    for emp_id, dates in employee_transitions.items():
        for date, transitions in dates.items():
            # Extract just the regions in sequence
            regions = [t['from'] for t in transitions]
            if transitions:  # Add the last destination
                regions.append(transitions[-1]['to'])
            
            # Extract sequences of length seq_length
            for i in range(len(regions) - seq_length + 1):
                seq = tuple(regions[i:i+seq_length])
                movement_sequences[seq] += 1
    
    # Get top movement sequences
    top_sequences = sorted(movement_sequences.items(), key=lambda x: x[1], reverse=True)[:20]
    
    return {
        'employee_transitions': dict(employee_transitions),
        'transition_counts': dict(transition_counts),
        'valid_transitions': valid_transitions,
        'invalid_transitions': invalid_transitions,
        'top_transitions': top_transitions[:20],  # Top 20 transitions
        'top_invalid_transitions': top_invalid[:10],  # Top 10 invalid transitions
        'employee_patterns': employee_patterns,
        'hub_regions': hub_regions[:10],  # Top 10 hub regions
        'top_sequences': top_sequences  # Top movement sequences
    }