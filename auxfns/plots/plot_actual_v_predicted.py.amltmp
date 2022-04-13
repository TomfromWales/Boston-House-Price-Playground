def plot_actual_v_predicted(df=None,feat=None,nbins=None,actual=None,pred=None,weight=None,title='<b>AvE</b>'):
    
  # -- Setup -------

  # packages
  from plotly.subplots import make_subplots
  import plotly.graph_objects as go
  import pandas as pd
  import numpy as np

  # -- Data Prep -------

  # take a copy
  inner_df = df.copy()

  # Band feat if nbins is provided
  if nbins is not None:
    inner_df[feat] = pd.Categorical(pd.qcut(inner_df[feat], q=nbins,duplicates="drop"))

  # Weights
  if weight is None:
    weight = "weight_temp"
    inner_df[weight] = np.ones(len(inner_df))    

  # Summarise by bucket
  inner_summary = inner_df.groupby([feat]).agg({actual: ['mean'], pred: ['mean'], weight: ['sum']}).reset_index()
  inner_summary.columns = ['feat', 'actuals_mean', 'preds_mean','weights_sum']

  # -- Plot -------

  # Plot prep
  actuals = inner_summary["actuals_mean"]
  preds = inner_summary["preds_mean"]
  weights = inner_summary["weights_sum"]
  x = pd.Categorical(inner_summary["feat"]).astype('str')

  # Create figure with secondary y-axis
  fig = make_subplots(specs=[[{"secondary_y": True}]])

  # Add traces
  fig.add_trace(
      go.Bar(x=x, y=weights, name="weights", marker_color="darkslateblue"),
      secondary_y=False,
  )  
  fig.add_trace(
      go.Scatter(x=x, y=actuals, name="actuals", marker_color="darkmagenta"),
      secondary_y=True,
  )
  fig.add_trace(
      go.Scatter(x=x, y=preds, name="preds", marker_color="darkcyan"),
      secondary_y=True,
  )

  # Window dressing
  fig.update_layout(
      title_text=title,    
      xaxis={'title':f"<b>{feat}</b>",'type':'category'},   
      yaxis={'range': [0,max(weights)*3], 'showgrid': False, 'title': "<b>bars</b>", 'side': "right"},
      yaxis2={'range': [min(preds.append(actuals))*0.25,max(preds.append(actuals))*1.17], 'title': "<b>lines</b>", 'side': "left"},
  )

  return fig