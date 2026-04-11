### SSL Methods with ResNet1D Backbone

Evaluation results on 90:10 train/test split.

<table>
  <thead>
    <tr>
      <th rowspan="2">Evaluation</th>
      <th colspan="4">SimCLR</th>
      <th colspan="4">SimCLRv2</th>
      <th colspan="4">BYOL</th>
      <th colspan="4">TNC</th>
    </tr>
    <tr>
      <th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1_w</th>
      <th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1_w</th>
      <th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1_w</th>
      <th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1_w</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>linear_probe</td>
      <td>0.6095</td><td>0.6095</td><td>0.6095</td><td>0.6086</td>
      <td>0.7267</td><td>0.7262</td><td>0.7267</td><td>0.7260</td>
      <td>0.6416</td><td>0.6462</td><td>0.6416</td><td>0.6410</td>
      <td>0.7402</td><td>0.7402</td><td>0.7402</td><td>0.7398</td>
    </tr>
    <tr>
      <td>shallow_MLP</td>
      <td>0.9011</td><td>0.9020</td><td>0.9011</td><td>0.9010</td>
      <td>0.9024</td><td>0.9034</td><td>0.9024</td><td>0.9025</td>
      <td>0.8186</td><td>0.8213</td><td>0.8186</td><td>0.8186</td>
      <td>0.9541</td><td>0.9543</td><td>0.9541</td><td>0.9541</td>
    </tr>
    <tr>
      <td>shallow_SVM</td>
      <td>0.5817</td><td>0.5821</td><td>0.5817</td><td>0.5793</td>
      <td>0.7039</td><td>0.7044</td><td>0.7039</td><td>0.7023</td>
      <td>0.5871</td><td>0.5942</td><td>0.5871</td><td>0.5843</td>
      <td>0.7036</td><td>0.7032</td><td>0.7036</td><td>0.7022</td>
    </tr>
    <tr>
      <td>shallow_DTree</td>
      <td>0.6758</td><td>0.6837</td><td>0.6758</td><td>0.6778</td>
      <td>0.7157</td><td>0.7231</td><td>0.7157</td><td>0.7171</td>
      <td>0.7625</td><td>0.7710</td><td>0.7625</td><td>0.7647</td>
      <td>0.8417</td><td>0.8432</td><td>0.8417</td><td>0.8421</td>
    </tr>
    <tr>
      <td>shallow_RF</td>
      <td>0.8991</td><td>0.8998</td><td>0.8991</td><td>0.8989</td>
      <td>0.9016</td><td>0.9022</td><td>0.9016</td><td>0.9015</td>
      <td>0.8635</td><td>0.8641</td><td>0.8635</td><td>0.8633</td>
      <td>0.9615</td><td>0.9617</td><td>0.9615</td><td>0.9615</td>
    </tr>
    <tr>
      <td>knn_k1</td>
      <td>0.9098</td><td>—</td><td>—</td><td>—</td>
      <td>0.9070</td><td>—</td><td>—</td><td>—</td>
      <td>0.8311</td><td>—</td><td>—</td><td>—</td>
      <td>0.9674</td><td>—</td><td>—</td><td>—</td>
    </tr>
    <tr>
      <td>knn_k5</td>
      <td>0.8875</td><td>—</td><td>—</td><td>—</td>
      <td>0.8828</td><td>—</td><td>—</td><td>—</td>
      <td>0.8141</td><td>—</td><td>—</td><td>—</td>
      <td>0.9541</td><td>—</td><td>—</td><td>—</td>
    </tr>
    <tr>
      <td>knn_k20</td>
      <td>0.8442</td><td>—</td><td>—</td><td>—</td>
      <td>0.8400</td><td>—</td><td>—</td><td>—</td>
      <td>0.7550</td><td>—</td><td>—</td><td>—</td>
      <td>0.9305</td><td>—</td><td>—</td><td>—</td>
    </tr>
    <tr>
      <td>fine_tune</td>
      <td>0.9833</td><td>0.9834</td><td>0.9833</td><td>0.9833</td>
      <td>0.9821</td><td>0.9822</td><td>0.9821</td><td>0.9821</td>
      <td>0.9868</td><td>0.9868</td><td>0.9868</td><td>0.9868</td>
      <td>0.9840</td><td>0.9841</td><td>0.9840</td><td>0.9840</td>
    </tr>
  </tbody>
</table>

> `—` indicates metric not applicable for that evaluator. Split: 90:10 for all rows.
