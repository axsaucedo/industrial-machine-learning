# Deep Learning with Recurrent Neural Networks

## [View All Presentation](https://axsauze.github.io/industrial-machine-learning/#/)

Presentation briefly introducing deep learning and how to apply a specific subset of deep learning, recurrent neural networks, to solve real world problems.

#### Slides: [http://axsauze.github.io/industrial-machine-learning/](http://axsauze.github.io/industrial-machine-learning/)
#### Slides Repo: [https://github.com/axsauze/crypto_ml](https://github.com/axsauze/crypto_ml)

Topics covered:

* Overview of presentation
* Machine learning expectation and reality
* Challenges of large scale, industry-ready machine learning
* Let's build a startup: CryptoML Limited
* Building the machine learning models
    * Starting out: Linear Regression
    * Exploring: Trying new models
    * Next level: Deep Learning
* Stepping up our deployment
    * Dockerising our codebase
    * Using conda / pypi to build
* Serving for the masses
    * Building a manager-worker architecture
    * Using celery and rabbitmq to account for this
* Taking it to the professional levels
    * Stepping up the game with Kubernetes
    * Using other devops tools: ELK Monitoring, etc

## Running Presentation

You can also run the presentation on a local web server. Clone this repository and run the presentation like so:

```
npm install
grunt serve
```

The presentation can now be accessed on `localhost:8080`. Note that this web application is configured to bind to hostname `0.0.0.0`, which means that once the Grunt server is running, it will be accessible from external hosts as well (using the current host's public IP address).

## Credits

A lot of the slides are re-used from a talk I did last year with [Donald Whyte](https://github.com/DonaldWhyte) in Moscow!

[![Moscow Python](https://img.youtube.com/vi/1GIBqPDzgwk/0.jpg)](https://www.youtube.com/watch?v=1GIBqPDzgwk)
