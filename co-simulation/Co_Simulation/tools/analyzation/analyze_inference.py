import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import scipy.stats as stats


def create_distplot(all_hist_data, group_labels, colors, pen_rate, name):
    #assert len(all_hist_data[) == len(group_labels) == len(colors)
    #assert len(all_hist_data) == len(pen_rates)

    all_data = all_hist_data

    means = [np.mean(x) for x in all_data]  # Calculate mean values
    #round the means to 2 decimal places
    means = [round(x, 2) for x in means]
    fig = ff.create_distplot(all_data, group_labels, show_hist=False, colors=colors, show_rug=False)

    #for i, trace in enumerate(fig['data']):
    #    trace.update(line=dict(width=2, dash=all_indicators[i % len(all_indicators)]))

    # Calculate KDE for each dataset and evaluate at the mean
    kde_values = []
    for data, mean in zip(all_data, means):
        # Create a Gaussian kernel density estimator
        kde = stats.gaussian_kde(data)
        # Evaluate the density at the mean
        kde_value_at_mean = kde(mean)
        kde_values.append(kde_value_at_mean)

    # Add mean lines
    for i, mean in enumerate(means):
        fig.add_shape(type="line",
                    x0=mean,
                    y0=0,
                    x1=mean,
                    y1=kde_values[i].item(),
                    line=dict(
                        color=colors[i],
                        width=2,
                        dash="dash",
                    ))

    # Update layout for the plot background and frame
    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', legend=dict(
            x=0,  # x=0 and y=1 places the legend at the top left
            y=1.1,
            traceorder="normal",
            orientation="h",
            font=dict(
                family="sans-serif",
                size=16,
                color="black"
            ),
            bgcolor="rgba(255,255,255,0.5)",  # Slightly transparent white background
            bordercolor="Black",
            borderwidth=1,
        )
    )
    # add bolt annotation
    fig.add_annotation(x=0.9,y=2.2, text=f'p={pen_rate}%', showarrow=False, font=dict(size=22, color='black'))
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror='allticks',
                    tickmode='array', #tickvals=[0,*means,1], ticktext=[0, means[0], means[1], means[2], means[3], 1],
                    range=[min(0, min(means)-0.1), max(1, max(means)+0.1)],
                    tickfont=dict(size=18))
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror='allticks', showticklabels=False)
    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
    # save with tight layout
    fig.write_image(f'{name}.png')
    # also save the fig as html
    fig.write_html(f'{name}.html')
    # also save as svg file
    fig.write_image(f'{name}.svg')



def analyze_detections(results_paths, pen_rates):
    # load the results
    assert len(results_paths) == len(pen_rates)
    all_hist_data = []
    all_group_labels = []
    real_group_label_matching = {
        '3d_detection': 'MonoCon',
        'emulation_detection': 'Emulated MonoCon',
        'kitti_detection': 'KITTI Criteria',
        'distance_detection': 'Distance Criteria'
    }
    for results_path, pen_rate in zip(results_paths, pen_rates):
        with open(results_path, 'rb') as f:
            results = pickle.load(f) # dict with keys of the inference results + simulation_time + fco_id

        # get the smallest time step
        time_steps = list(set([result['simulation_time'] for result in results.values()]))

        detection_types = list(list(results.values())[0].keys())
        detection_types.remove('simulation_time')
        detection_types.remove('fco_id')
        detection_types.remove('all_vehicles')

        # convert the results dict into pandas dataframe
        results_df = pd.DataFrame(results)
        results_df = results_df.T

        relative_detections = {detection_type: [] for detection_type in detection_types}

        for timestep in time_steps:
            timestep_df = results_df[results_df['simulation_time'] == timestep]
            total_vehicles = timestep_df['all_vehicles'].values[0]
            for detection_type in detection_types:
                current_detection_list = relative_detections[detection_type]
                all_fcos = timestep_df['fco_id'].values.tolist()
                detected_vehicles = timestep_df[detection_type].values.tolist()
                # flatten the list of lists
                detected_vehicles = [item for sublist in detected_vehicles for item in sublist]
                all_detections = list(set(all_fcos + detected_vehicles))
                print(f'detection_type: {detection_type}, len_detected_vehicles: {len(detected_vehicles)}')
                if detection_type == 'emulation_detection':
                    current_detection_list.append((len(all_detections)-1)/len(total_vehicles))
                else:
                    current_detection_list.append(len(all_detections)/len(total_vehicles))
                relative_detections[detection_type] = current_detection_list
        hist_data = [relative_detections[detection_type] for detection_type in detection_types]
        colors = ['blue', 'green', 'orange', 'purple']
        real_group_labels = [real_group_label_matching[label] for label in detection_types]
        create_distplot(hist_data, real_group_labels, colors,pen_rate, f'distplot_{pen_rate}')
        
        
        #hist_data = [relative_detections[detection_type] for detection_type in detection_types]
        #group_labels = [f'{detection_type}_{pen_rate}' for detection_type in relative_detections.keys()]
        #all_hist_data.append(hist_data)
        #all_group_labels.extend(group_labels)

    
    pen_rate_indicators = ['solid', 'dash']
    #real_group_labels = [real_group_label_matching[label] for label in group_labels]
    real_group_labels = all_group_labels

    

    

    # create distplot with showing





if __name__ == "__main__":
    results_path = ['inference_results_10.pkl', 'inference_results.pkl']
    pen_rates = [10, 20]
    analyze_detections(results_path, pen_rates)