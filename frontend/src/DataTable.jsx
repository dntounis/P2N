import React from 'react';

const DataTable = ({ data }) => {
  // If data is null or undefined
  if (!data) return null;

  // Function to extract data_series from the parsed JSON
  const extractDataSeries = (resultObj) => {
    try {
      if (resultObj?.p2n?.gt_parse?.panels) {
        return resultObj.p2n.gt_parse.panels.flatMap(p => p.data_series || []);
      }
      // If it's the raw format directly
      if (resultObj?.panels) {
        return resultObj.panels.flatMap(p => p.data_series || []);
      }
    } catch (e) {
      return [];
    }
    return [];
  };

  const dataSeries = extractDataSeries(data);

  if (!dataSeries || dataSeries.length === 0) {
    return (
      <div className="empty-state">
        <p>No tabular data could be parsed from the model output.</p>
        <p style={{fontSize: '0.85rem', opacity: 0.7, marginTop: '0.5rem'}}>
          (Check the Raw JSON tab to see what the model generated)
        </p>
      </div>
    );
  }

  return (
    <div className="data-tables-container">
      {dataSeries.map((series, idx) => {
        const values = series.values || series; // sometimes it's just an array, sometimes {type: '...', values: [...]}
        const type = series.type || 'Data Series';
        
        if (!Array.isArray(values) || values.length === 0) return null;

        // Determine if it's an array of objects [{x: 1, y: 2}] or flat array [1.0, 2.0]
        const isObjectArray = typeof values[0] === 'object' && values[0] !== null;
        
        let headers = [];
        if (isObjectArray) {
          // get all unique keys
          const keySet = new Set();
          values.forEach(v => Object.keys(v).forEach(k => keySet.add(k)));
          headers = Array.from(keySet);
        } else {
          headers = ['Value'];
        }

        return (
          <div key={idx} className="table-wrapper">
            <h3>{type} {idx + 1}</h3>
            <div className="table-scroll">
              <table className="interactive-table">
                <thead>
                  <tr>
                    {headers.map(h => <th key={h}>{h}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {values.map((row, rowIdx) => (
                    <tr key={rowIdx}>
                      {isObjectArray 
                        ? headers.map(h => <td key={h}>{row[h] !== undefined ? row[h] : '-'}</td>)
                        : <td>{row}</td>
                      }
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default DataTable;
