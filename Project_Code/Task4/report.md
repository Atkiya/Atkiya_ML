### SSL Methods with ResNet1D Backbone


<table>
  <thead>
    <tr>
      <th rowspan="2">Evaluation</th>
      <th colspan="4">SimCLRv1</th>
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
      <td>Linear Probe</td>
      <td>0.6095</td><td>0.6095</td><td>0.6095</td><td>0.6086</td>
      <td>0.7267</td><td>0.7262</td><td>0.7267</td><td>0.7260</td>
      <td>0.6416</td><td>0.6462</td><td>0.6416</td><td>0.6410</td>
      <td>0.7402</td><td>0.7402</td><td>0.7402</td><td>0.7398</td>
    </tr>
    <tr>
      <td>MLP</td>
      <td>0.9011</td><td>0.9020</td><td>0.9011</td><td>0.9010</td>
      <td>0.9024</td><td>0.9034</td><td>0.9024</td><td>0.9025</td>
      <td>0.8186</td><td>0.8213</td><td>0.8186</td><td>0.8186</td>
      <td>0.9541</td><td>0.9543</td><td>0.9541</td><td>0.9541</td>
    </tr>
    <tr>
      <td>SVM</td>
      <td>0.5817</td><td>0.5821</td><td>0.5817</td><td>0.5793</td>
      <td>0.7039</td><td>0.7044</td><td>0.7039</td><td>0.7023</td>
      <td>0.5871</td><td>0.5942</td><td>0.5871</td><td>0.5843</td>
      <td>0.7036</td><td>0.7032</td><td>0.7036</td><td>0.7022</td>
    </tr>
    <tr>
      <td>DT</td>
      <td>0.6758</td><td>0.6837</td><td>0.6758</td><td>0.6778</td>
      <td>0.7157</td><td>0.7231</td><td>0.7157</td><td>0.7171</td>
      <td>0.7625</td><td>0.7710</td><td>0.7625</td><td>0.7647</td>
      <td>0.8417</td><td>0.8432</td><td>0.8417</td><td>0.8421</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.8991</td><td>0.8998</td><td>0.8991</td><td>0.8989</td>
      <td>0.9016</td><td>0.9022</td><td>0.9016</td><td>0.9015</td>
      <td>0.8635</td><td>0.8641</td><td>0.8635</td><td>0.8633</td>
      <td>0.9615</td><td>0.9617</td><td>0.9615</td><td>0.9615</td>
    </tr>
    <tr>
      <td>K = 1</td>
      <td>0.9098</td><td>—</td><td>—</td><td>—</td>
      <td>0.9070</td><td>—</td><td>—</td><td>—</td>
      <td>0.8311</td><td>—</td><td>—</td><td>—</td>
      <td>0.9674</td><td>—</td><td>—</td><td>—</td>
    </tr>
    <tr>
      <td>K = 5</td>
      <td>0.8875</td><td>—</td><td>—</td><td>—</td>
      <td>0.8828</td><td>—</td><td>—</td><td>—</td>
      <td>0.8141</td><td>—</td><td>—</td><td>—</td>
      <td>0.9541</td><td>—</td><td>—</td><td>—</td>
    </tr>
    <tr>
      <td>K = 20</td>
      <td>0.8442</td><td>—</td><td>—</td><td>—</td>
      <td>0.8400</td><td>—</td><td>—</td><td>—</td>
      <td>0.7550</td><td>—</td><td>—</td><td>—</td>
      <td>0.9305</td><td>—</td><td>—</td><td>—</td>
    </tr>
    <tr>
      <td>Fine Tune</td>
      <td>0.9833</td><td>0.9834</td><td>0.9833</td><td>0.9833</td>
      <td>0.9821</td><td>0.9822</td><td>0.9821</td><td>0.9821</td>
      <td>0.9868</td><td>0.9868</td><td>0.9868</td><td>0.9868</td>
      <td>0.9840</td><td>0.9841</td><td>0.9840</td><td>0.9840</td>
    </tr>
  </tbody>
</table>

### SSL Methods with DNN Encoder


<table>
  <thead>
    <tr>
      <th rowspan="2">Evaluation</th>
      <th colspan="4">SimCLRv1</th>
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
      <td>Linear Probe</td>
      <td>0.4312</td><td>0.4307</td><td>0.4312</td><td>0.4277</td>
      <td>0.5027</td><td>0.5050</td><td>0.5027</td><td>0.5021</td>
      <td>0.3375</td><td>0.3378</td><td>0.3375</td><td>0.3325</td>
      <td>0.4581</td><td>0.4629</td><td>0.4581</td><td>0.4577</td>
    </tr>
    <tr>
      <td>MLP</td>
      <td>0.7341</td><td>0.7373</td><td>0.7341</td><td>0.7336</td>
      <td>0.7377</td><td>0.7400</td><td>0.7377</td><td>0.7374</td>
      <td>0.4622</td><td>0.4714</td><td>0.4622</td><td>0.4601</td>
      <td>0.7496</td><td>0.7550</td><td>0.7496</td><td>0.7492</td>
    </tr>
    <tr>
      <td>SVM</td>
      <td>0.4168</td><td>0.4158</td><td>0.4168</td><td>0.4076</td>
      <td>0.4817</td><td>0.4838</td><td>0.4817</td><td>0.4782</td>
      <td>0.3307</td><td>0.3311</td><td>0.3307</td><td>0.3185</td>
      <td>0.4453</td><td>0.4513</td><td>0.4453</td><td>0.4420</td>
    </tr>
    <tr>
      <td>DT</td>
      <td>0.5128</td><td>0.5155</td><td>0.5128</td><td>0.5135</td>
      <td>0.5242</td><td>0.5258</td><td>0.5242</td><td>0.5242</td>
      <td>0.2575</td><td>0.2580</td><td>0.2575</td><td>0.2575</td>
      <td>0.4957</td><td>0.4976</td><td>0.4957</td><td>0.4959</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.7373</td><td>0.7416</td><td>0.7373</td><td>0.7375</td>
      <td>0.7385</td><td>0.7419</td><td>0.7385</td><td>0.7385</td>
      <td>0.3880</td><td>0.3910</td><td>0.3880</td><td>0.3851</td>
      <td>0.7062</td><td>0.7127</td><td>0.7062</td><td>0.7070</td>
    </tr>
    <tr>
      <td>K = 1</td>
      <td>0.7529</td><td>—</td><td>—</td><td>—</td>
      <td>0.7354</td><td>—</td><td>—</td><td>—</td>
      <td>0.3198</td><td>—</td><td>—</td><td>—</td>
      <td>0.7242</td><td>—</td><td>—</td><td>—</td>
    </tr>
    <tr>
      <td>K = 5</td>
      <td>0.7279</td><td>—</td><td>—</td><td>—</td>
      <td>0.7141</td><td>—</td><td>—</td><td>—</td>
      <td>0.3062</td><td>—</td><td>—</td><td>—</td>
      <td>0.6922</td><td>—</td><td>—</td><td>—</td>
    </tr>
    <tr>
      <td>K = 20</td>
      <td>0.6804</td><td>—</td><td>—</td><td>—</td>
      <td>0.6697</td><td>—</td><td>—</td><td>—</td>
      <td>0.2998</td><td>—</td><td>—</td><td>—</td>
      <td>0.6435</td><td>—</td><td>—</td><td>—</td>
    </tr>
    <tr>
      <td>Fine Tune</td>
      <td>0.8669</td><td>0.8676</td><td>0.8669</td><td>0.8669</td>
      <td>0.8657</td><td>0.8666</td><td>0.8657</td><td>0.8657</td>
      <td>0.8553</td><td>0.8566</td><td>0.8553</td><td>0.8554</td>
      <td>0.9069</td><td>0.9075</td><td>0.9069</td><td>0.9069</td>
    </tr>
  </tbody>
</table>

### SSL Methods with Custom CNN 1D Encoder


<table>
  <thead>
    <tr>
      <th rowspan="2">Evaluation</th>
      <th colspan="4">SimCLRv1</th>
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
      <td>Linear Probe</td>
      <td>0.5895</td><td>0.5889</td><td>0.5895</td><td>0.5885</td>
      <td>0.6875</td><td>0.6864</td><td>0.6875</td><td>0.6863</td>
      <td>0.6645</td><td>0.6694</td><td>0.6645</td><td>0.6641</td>
      <td>0.7424</td><td>0.7418</td><td>0.7424</td><td>0.7418</td>
    </tr>
    <tr>
      <td>MLP</td>
      <td>0.8875</td><td>0.8884</td><td>0.8875</td><td>0.8875</td>
      <td>0.8823</td><td>0.8831</td><td>0.8823</td><td>0.8823</td>
      <td>0.8135</td><td>0.8157</td><td>0.8135</td><td>0.8131</td>
      <td>0.9506</td><td>0.9508</td><td>0.9506</td><td>0.9506</td>
    </tr>
    <tr>
      <td>SVM</td>
      <td>0.5608</td><td>0.5588</td><td>0.5608</td><td>0.5568</td>
      <td>0.6667</td><td>0.6651</td><td>0.6667</td><td>0.6643</td>
      <td>0.6172</td><td>0.6251</td><td>0.6172</td><td>0.6145</td>
      <td>0.7005</td><td>0.6992</td><td>0.7005</td><td>0.6983</td>
    </tr>
    <tr>
      <td>DT</td>
      <td>0.6275</td><td>0.6314</td><td>0.6275</td><td>0.6287</td>
      <td>0.6485</td><td>0.6554</td><td>0.6485</td><td>0.6496</td>
      <td>0.7724</td><td>0.7763</td><td>0.7724</td><td>0.7726</td>
      <td>0.8232</td><td>0.8249</td><td>0.8232</td><td>0.8237</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.8729</td><td>0.8745</td><td>0.8729</td><td>0.8727</td>
      <td>0.8859</td><td>0.8867</td><td>0.8859</td><td>0.8857</td>
      <td>0.8659</td><td>0.8662</td><td>0.8659</td><td>0.8656</td>
      <td>0.9558</td><td>0.9560</td><td>0.9558</td><td>0.9558</td>
    </tr>
    <tr>
      <td>K = 1</td>
      <td>0.9162</td><td>—</td><td>—</td><td>—</td>
      <td>0.9114</td><td>—</td><td>—</td><td>—</td>
      <td>0.8325</td><td>—</td><td>—</td><td>—</td>
      <td>0.9642</td><td>—</td><td>—</td><td>—</td>
    </tr>
    <tr>
      <td>K = 5</td>
      <td>0.8840</td><td>—</td><td>—</td><td>—</td>
      <td>0.8798</td><td>—</td><td>—</td><td>—</td>
      <td>0.8137</td><td>—</td><td>—</td><td>—</td>
      <td>0.9528</td><td>—</td><td>—</td><td>—</td>
    </tr>
    <tr>
      <td>K = 20</td>
      <td>0.8360</td><td>—</td><td>—</td><td>—</td>
      <td>0.8376</td><td>—</td><td>—</td><td>—</td>
      <td>0.7582</td><td>—</td><td>—</td><td>—</td>
      <td>0.9282</td><td>—</td><td>—</td><td>—</td>
    </tr>
    <tr>
      <td>Fine Tune</td>
      <td>0.9812</td><td>0.9813</td><td>0.9812</td><td>0.9812</td>
      <td>0.9836</td><td>0.9836</td><td>0.9836</td><td>0.9836</td>
      <td>0.9892</td><td>0.9892</td><td>0.9892</td><td>0.9892</td>
      <td>0.9868</td><td>0.9868</td><td>0.9868</td><td>0.9868</td>
    </tr>
  </tbody>
</table>

> `—` indicates metric not applicable for that evaluator. Split: 90:10 for all rows.

### Accuracy Comparison Across SSL Methods and Encoder Architectures


<table>
  <thead>
    <tr>
      <th rowspan="2">Evaluation</th>
      <th colspan="3">SimCLRv1</th>
      <th colspan="3">SimCLRv2</th>
      <th colspan="3">BYOL</th>
      <th colspan="3">TNC</th>
    </tr>
    <tr>
      <th>ResNet1D</th><th>DNN</th><th>Custom CNN</th>
      <th>ResNet1D</th><th>DNN</th><th>Custom CNN</th>
      <th>ResNet1D</th><th>DNN</th><th>Custom CNN</th>
      <th>ResNet1D</th><th>DNN</th><th>Custom CNN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Linear Probe</td>
      <td>0.6095</td><td>0.4312</td><td>0.5895</td>
      <td>0.7267</td><td>0.5027</td><td>0.6875</td>
      <td>0.6416</td><td>0.3375</td><td>0.6645</td>
      <td>0.7402</td><td>0.4581</td><td>0.7424</td>
    </tr>
    <tr>
      <td>MLP</td>
      <td>0.9011</td><td>0.7341</td><td>0.8875</td>
      <td>0.9024</td><td>0.7377</td><td>0.8823</td>
      <td>0.8186</td><td>0.4622</td><td>0.8135</td>
      <td>0.9541</td><td>0.7496</td><td>0.9506</td>
    </tr>
    <tr>
      <td>SVM</td>
      <td>0.5817</td><td>0.4168</td><td>0.5608</td>
      <td>0.7039</td><td>0.4817</td><td>0.6667</td>
      <td>0.5871</td><td>0.3307</td><td>0.6172</td>
      <td>0.7036</td><td>0.4453</td><td>0.7005</td>
    </tr>
    <tr>
      <td>DT</td>
      <td>0.6758</td><td>0.5128</td><td>0.6275</td>
      <td>0.7157</td><td>0.5242</td><td>0.6485</td>
      <td>0.7625</td><td>0.2575</td><td>0.7724</td>
      <td>0.8417</td><td>0.4957</td><td>0.8232</td>
    </tr>
    <tr>
      <td>RF</td>
      <td>0.8991</td><td>0.7373</td><td>0.8729</td>
      <td>0.9016</td><td>0.7385</td><td>0.8859</td>
      <td>0.8635</td><td>0.3880</td><td>0.8659</td>
      <td>0.9615</td><td>0.7062</td><td>0.9558</td>
    </tr>
    <tr>
      <td>K = 1</td>
      <td>0.9098</td><td>0.7529</td><td>0.9162</td>
      <td>0.9070</td><td>0.7354</td><td>0.9114</td>
      <td>0.8311</td><td>0.3198</td><td>0.8325</td>
      <td>0.9674</td><td>0.7242</td><td>0.9642</td>
    </tr>
    <tr>
      <td>K = 5</td>
      <td>0.8875</td><td>0.7279</td><td>0.8840</td>
      <td>0.8828</td><td>0.7141</td><td>0.8798</td>
      <td>0.8141</td><td>0.3062</td><td>0.8137</td>
      <td>0.9541</td><td>0.6922</td><td>0.9528</td>
    </tr>
    <tr>
      <td>K = 20</td>
      <td>0.8442</td><td>0.6804</td><td>0.8360</td>
      <td>0.8400</td><td>0.6697</td><td>0.8376</td>
      <td>0.7550</td><td>0.2998</td><td>0.7582</td>
      <td>0.9305</td><td>0.6435</td><td>0.9282</td>
    </tr>
    <tr>
      <td>Fine Tune</td>
      <td>0.9833</td><td>0.8669</td><td>0.9812</td>
      <td>0.9821</td><td>0.8657</td><td>0.9836</td>
      <td>0.9868</td><td>0.8553</td><td>0.9892</td>
      <td>0.9840</td><td>0.9069</td><td>0.9868</td>
    </tr>
  </tbody>
</table>

### Summary

Among all evaluated SSL methods and encoder architectures, **TNC (Temporal Neighbourhood 
Coding) paired with the ResNet1D backbone** consistently delivers the strongest performance 
across nearly all evaluation strategies, achieving accuracy scores of **0.9674** (KNN, K=1), 
**0.9615** (Random Forest), and **0.9541** (MLP). Unlike fine-tuning-dependent methods, TNC 
produces inherently richer feature representations, as evidenced by its dominance in linear 
probing and non-parametric evaluators where the frozen encoder is directly tested. While BYOL 
with a Custom CNN encoder achieves a marginally higher fine-tune accuracy of **0.9892**, this 
metric reflects the downstream classifier's strength rather than the quality of the learned 
representations. The DNN encoder proves to be the weakest backbone across all SSL methods, 
often trailing by **15–30 percentage points**, further reinforcing that architectural choice 
plays a critical role alongside the SSL training objective. Overall, TNC with ResNet1D offers 
the best balance of representation quality, generalizability, and consistent performance, 
making it the most suitable choice for this task.
