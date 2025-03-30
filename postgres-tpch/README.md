# TPC-H in dockerized postgres

Postgres database initialised with the [TPC-H](http://www.tpc.org/tpch/) dataset
at a specified scale factor.

* Build the docker image: `docker build --tag postgres-tpch:1.0 .`
* Create and run the container: `docker run --name postgres-tpch-001 -e POSTGRES_PASSWORD=postgres -e SCALE_FACTOR=0.01 postgres-tpch:1.0`
* Stop the container:`docker stop postgres-tpch-001`
* Remove the container: `docker rm postgres-tpch-001`

You can modify the environment parameter `SCALE_FACTOR` to generate a bigger or
smaller TPC-H dataset.
