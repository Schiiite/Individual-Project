import plotly
import plotly.graph_objs as go


def make_bar_plot(x_axis, y_axis, plot_name, x_axis_name, y_axis_name):
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name = plot_name,
            x = x_axis,
            y = y_axis,
        )
    )
    fig.update_layout(
        title=plot_name,
        xaxis=dict(title=x_axis_name),
        yaxis=dict(title=y_axis_name),
        font=dict(size=25),
    )
    fig.show()


def make_scatter_plot(x_axis, y_axis, plot_name, x_axis_name, y_axis_name):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            name = plot_name,
            x = x_axis,
            y = y_axis,
            # mode = "markers + lines",
        )
    )
    fig.update_layout(
        title = plot_name,
        xaxis=dict(title=x_axis_name),
        yaxis=dict(title=y_axis_name),
        font=dict(size=25),
    )
    fig.show()


def add_scatters(fig, x_axis, y_axis, l_name):
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=y_axis,
            name=l_name,
            mode="markers + lines",
        )
        fig.update_layout(
            font=dict(size=25),
        )
    )


def fig_update(fig, plot_name, x_axis_name, y_axis_name):
    fig.update_layout(
        title=plot_name,
        xaxis=dict(title=x_axis_name),
        yaxis=dict(title=y_axis_name),
        font=dict(size=25),
    )
    # fig.update_xaxes(rangeslider_visible=True)
