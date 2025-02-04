import pandas as pd

def get_interaction_net_probabilities(df):
    """
    For calculating the handover probabilities between agent types in the agentminer interaction net
    E.g., p(role_2|role_1)

    Parameters:
        -----------
        df : pandas df
            The interaction log from the agentminer.

        Returns:
        --------
        transition_probabilities : dict
            Dictionary containing handover probabilities from each role to every other role.
    """
    # Convert end_timestamp to datetime
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], format='mixed')

    # Sort DataFrame by end_timestamp
    # df = df.sort_values(by=['case_id', 'end_timestamp'])

    # Group by case_id
    grouped = df.groupby('case_id')

    # Initialize transition count dictionary
    transition_counts = {}

    unique_agents = df['agent_id'].unique().tolist()
    unique_agents.insert(0, '-1')
    agent_counts = {agent: 0 for agent in unique_agents}

    # Iterate over groups
    for _, group in grouped:
        agents = group['agent_id'].tolist()
        # insert artificial agent at the beginning to represent start of process
        agents.insert(0, '-1')
        for i in range(len(agents) - 1):
            transition = (agents[i], agents[i+1])
            if transition in transition_counts:
                transition_counts[transition] += 1
            else:
                transition_counts[transition] = 1
            agent_counts[agents[i]] += 1

    # Calculate transition probabilities
    total_transitions = sum(transition_counts.values())
    
    # Initialize transition probabilities dictionary
    transition_probabilities = {}

    # Iterate over transitions
    for transition, count in transition_counts.items():
        agent_from, agent_to = transition
        if agent_from not in transition_probabilities:
            transition_probabilities[agent_from] = {}
        transition_probabilities[agent_from][agent_to] = count / agent_counts[agent_from]

    return transition_probabilities


def calculate_agent_handover_probabilities(df):
    """
    For calculating the handover probabilities between agents 
    E.g., p(agent_2|agent_1)

    Parameters:
        -----------
        df : pandas df
            The training log.

        Returns:
        --------
        transition_probabilities : dict
            Dictionary containing handover probabilities from each agent to every other agent.
    """
    # Convert end_timestamp to datetime
    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'], format='mixed')

    # Sort DataFrame by end_timestamp
    df = df.sort_values(by=['case_id', 'end_timestamp'])

    # Group by case_id
    grouped = df.groupby('case_id')

    # Initialize transition count dictionary
    transition_counts = {}

    agent_counts = {agent: 0 for agent in df['agent'].unique()}

    # Iterate over groups
    for _, group in grouped:
        agents = group['agent'].tolist()
        for i in range(len(agents) - 1):
            transition = (agents[i], agents[i+1])
            if transition in transition_counts:
                transition_counts[transition] += 1
            else:
                transition_counts[transition] = 1
            agent_counts[agents[i]] += 1

    # Calculate transition probabilities
    total_transitions = sum(transition_counts.values())
    
    # Initialize transition probabilities dictionary
    transition_probabilities = {}

    # Iterate over transitions
    for transition, count in transition_counts.items():
        agent_from, agent_to = transition
        if agent_from not in transition_probabilities:
            transition_probabilities[agent_from] = {}
        transition_probabilities[agent_from][agent_to] = count / agent_counts[agent_from]

    return transition_probabilities


# def calculate_agent_handover_probabilities_per_activity(df):
#     """
#     For calculating the handover probabilities between agents per activity
#     E.g., p(agent_2|agent_1, activity_1)

#     Parameters:
#         -----------
#         df : pandas df
#             The training log.

#         Returns:
#         --------
#         transition_probabilities : dict
#             Dictionary containing handover probabilities from each agent to every other agent.
#     """
#     # Convert end_timestamp to datetime
#     df['end_timestamp'] = pd.to_datetime(df['end_timestamp'], format='mixed')

#     # Sort DataFrame by end_timestamp
#     df = df.sort_values(by=['case_id', 'end_timestamp'])

#     # Group by case_id
#     grouped = df.groupby('case_id')

#     # Initialize transition count dictionary
#     transition_counts = {}

#     # Initialize agent-activity counts dictionary
#     agent_activity_counts = {}

#     # Iterate over groups
#     for _, group in grouped:
#         agents = group['agent'].tolist()
#         activities = group['activity_name'].tolist()
        
#         for i in range(len(agents) - 1):
#             # Create transition tuple with (from_agent, activity, to_agent)
#             transition = (agents[i], activities[i], agents[i+1])
            
#             # Update transition counts
#             if transition in transition_counts:
#                 transition_counts[transition] += 1
#             else:
#                 transition_counts[transition] = 1
                
#             # Update agent-activity counts
#             agent_activity_key = (agents[i], activities[i])
#             if agent_activity_key in agent_activity_counts:
#                 agent_activity_counts[agent_activity_key] += 1
#             else:
#                 agent_activity_counts[agent_activity_key] = 1

#     # Initialize transition probabilities dictionary
#     transition_probabilities = {}

#     # Calculate transition probabilities
#     for transition, count in transition_counts.items():
#         agent_from, activity, agent_to = transition
        
#         # Create nested dictionaries if they don't exist
#         if agent_from not in transition_probabilities:
#             transition_probabilities[agent_from] = {}
#         if activity not in transition_probabilities[agent_from]:
#             transition_probabilities[agent_from][activity] = {}
            
#         # Calculate probability: count(from_agent, activity, to_agent) / count(from_agent, activity)
#         agent_activity_count = agent_activity_counts[(agent_from, activity)]
#         transition_probabilities[agent_from][activity][agent_to] = count / agent_activity_count

#     return transition_probabilities

def calculate_agent_handover_probabilities_per_activity(df):
    """
    For calculating the handover probabilities between agents per activity pair
    E.g., p(agent_2, activity_2 | agent_1, activity_1)

    Parameters:
        -----------
        df : pandas df
            The training log.

        Returns:
        --------
        transition_probabilities : dict
            Dictionary containing handover probabilities from each agent-activity pair 
            to every other agent-activity pair.
    """
    # Convert end_timestamp to datetime
    df['end_timestamp'] = pd.to_datetime(df['end_timestamp'], format='mixed')

    # Sort DataFrame by end_timestamp
    df = df.sort_values(by=['case_id', 'end_timestamp'])

    # Group by case_id
    grouped = df.groupby('case_id')
    
    # Initialize transition count dictionary
    transition_counts = {}

    # Initialize agent-activity counts dictionary
    agent_activity_counts = {}

    # Iterate over groups
    for _, group in grouped:
        agents = group['agent'].tolist()
        activities = group['activity_name'].tolist()
        
        for i in range(len(agents) - 1):
            # Create transition tuple with (from_agent, from_activity, to_agent, to_activity)
            transition = (agents[i], activities[i], agents[i+1], activities[i+1])
            
            # Update transition counts
            if transition in transition_counts:
                transition_counts[transition] += 1
            else:
                transition_counts[transition] = 1
                
            # Update agent-activity counts
            agent_activity_key = (agents[i], activities[i])
            if agent_activity_key in agent_activity_counts:
                agent_activity_counts[agent_activity_key] += 1
            else:
                agent_activity_counts[agent_activity_key] = 1

    # Initialize transition probabilities dictionary
    transition_probabilities = {}

    # Calculate transition probabilities
    for transition, count in transition_counts.items():
        agent_from, activity_from, agent_to, activity_to = transition
        
        # Create nested dictionaries if they don't exist
        if agent_from not in transition_probabilities:
            transition_probabilities[agent_from] = {}
        if activity_from not in transition_probabilities[agent_from]:
            transition_probabilities[agent_from][activity_from] = {}
        if agent_to not in transition_probabilities[agent_from][activity_from]:
            transition_probabilities[agent_from][activity_from][agent_to] = {}
            
        # Calculate probability: count(from_agent, from_activity, to_agent, to_activity) / count(from_agent, from_activity)
        agent_activity_count = agent_activity_counts[(agent_from, activity_from)]
        transition_probabilities[agent_from][activity_from][agent_to][activity_to] = count / agent_activity_count

    print(f"transition_probabilities: {transition_probabilities}")

    return transition_probabilities