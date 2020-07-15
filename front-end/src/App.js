import React, { useEffect, useState } from "react";
import { BrowserRouter, Switch, Route } from "react-router-dom";
import predictionService from "./services/predictions";
import Chart from "./components/charts";
import Header from "./components/Header";
import Home from "./components/pages/Home";

function App() {
  const [xLabel, setXLabel] = useState("");
  const [yLabel, setYLabel] = useState("");
  const [predict, setPredict] = useState("meal");
  const [xData, setXData] = useState([]);
  const [yData, setYData] = useState([]);
  const [dataType, setDataType] = useState("/train/predictions");
  useEffect(() => {
    predictionService.getAll(predict.concat(dataType)).then(response => {
      let data = Object.entries(response.data);
      setXLabel(data[0][0]);
      setXData(Object.values(data[0][1]));
      setYLabel(data[1][0]);
      setYData(Object.values(data[1][1]));
    });
  }, [predict, dataType]);
  const state = {
    labels: xData,
    datasets: [
      {
        label: yLabel,
        backgroundColor: "rgba(75,192,192,1)",
        borderColor: "rgba(0,0,0,1)",
        borderWidth: 2,
        data: yData
      }
    ]
  };
  const handleChange = (event, newDataType) => {
    if (dataType && newDataType) {
      setDataType(newDataType);
    }
  };
  const handleDownload = () => {
    window.open(`${predictionService.baseUrl}${predict}/predictions/download`);
  };
  return (
    <BrowserRouter>
      <Header handleChange={handleChange} dataType={dataType} />
      <Switch>
        <Route exact path="/" render={props => <Home {...props} />} />
        <Route
          exact
          path="/meal"
          render={props => {
            return (
              <Chart
                {...props}
                state={state}
                xLabel={xLabel}
                yLabel={yLabel}
                handleDownload={handleDownload}
                setPredict={setPredict}
                label="meal"
              />
            );
          }}
        />
        <Route
          exact
          path="/category"
          render={props => {
            return (
              <Chart
                {...props}
                state={state}
                xLabel={xLabel}
                yLabel={yLabel}
                handleDownload={handleDownload}
                setPredict={setPredict}
                label="category"
              />
            );
          }}
        />
        <Route
          exact
          path="/cuisine"
          render={props => {
            return (
              <Chart
                {...props}
                state={state}
                xLabel={xLabel}
                yLabel={yLabel}
                handleDownload={handleDownload}
                setPredict={setPredict}
                label="cuisine"
              />
            );
          }}
        />
        <Route
          exact
          path="/center"
          render={props => {
            return (
              <Chart
                {...props}
                state={state}
                xLabel={xLabel}
                yLabel={yLabel}
                handleDownload={handleDownload}
                setPredict={setPredict}
                label="center"
              />
            );
          }}
        />
      </Switch>
    </BrowserRouter>
  );
}

export default App;
