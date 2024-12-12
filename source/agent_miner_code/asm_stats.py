def read_entropia_results(entropia_output_file_name):
        precision = -0.99
        recall = -0.99
        with open(entropia_output_file_name, 'r') as entopia_file:
                lines = entopia_file.readlines()
                if len(lines)>1: 
                        precision_line_parts = lines[-2].split(' ')
                        print("precision_line_parts",precision_line_parts)
                        if len(precision_line_parts)>1:
                                precStr = precision_line_parts[1].strip('\n').strip('.')
                                if precStr.replace(".","").isnumeric():
                                        precision = float(precStr)
                        recall_line_parts = lines[-1].split(' ')
                        print("recall_line_parts",recall_line_parts)
                        if len(recall_line_parts)>1:
                                recallStr = recall_line_parts[1].strip('\n').strip('.')
                                if recallStr.replace(".","").isnumeric():
                                        recall = float(recallStr)
        return (precision,recall)
