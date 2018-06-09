<h1>Streamlined Machine Learning</h1>
<hr>
<strong>Streamlined Machine Learning</strong> is a set of robust functions and classes meant to streamline: preprocessing, model selection, and feature selection.
This package is build on top of <em>SciPy</em> and <em>sklearn</em>.

<h2>Basic Usage</h2>
By building a <code>Stream</code> object, you can specify a list of predefined objects the package manages, then you can <code>flow</code> through them each on default grid selection parameters or user defined parameters (denoted <code>params</code>).
Streams provided:
<ul>

<li><code>TransformationStream</code>, meant to flow through preprocessing techniques such as: scaling, normalizing, boxcox, binarization, pca, or kmeans aimed at returning a desired input dataset for model development.</li>

<li><code>ModelSelectionStream</code>, meant to flow through several predictive models to determine which is the best, these include: LinearRegression, SupportVectorRegressor, RandomForestRegressor, KNNRegressor, and others. You must specify whether your steam is a <em>regressor</em> or <em>classifier</em> stream (denoted <code>regressor=True</code> and <code>classifier=True</code> </li>

<li><code>FeatureSelectionStream</code>, meant to flow through several predictive models and algorithms to determine which subset of features is most predictive or representative of your dataset, these include: RandomForestFeatureImportance, LassoFeatureImportance, MixedSelection, and a technique to ensemble each named TOPSISFeatureRanking. You must specify whether your wish to ensemble and with what technique (denoted <code>ensemble=True</code> </li>

<ul></ul>
</ul>
