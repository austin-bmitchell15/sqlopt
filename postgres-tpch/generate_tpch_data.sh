#!/bin/bash
cd /tmp/2.18.0_rc2/dbgen
./dbgen -v -s $SCALE_FACTOR
export DSS_QUERY=./queries
for r in {1..100}; do
    for q in {1..22}; do
        ./qgen -c -s $SCALE_FACTOR -r ${r} $q > /generated_queries/query${q}.${r}.sql
    done
done
# Remove the final '|' delimiter from every line otherwise data cannot be
# imported in Postgres
sed -i 's/|$//' *.tbl
mkdir /tmp/tpchdata
mv *.tbl /tmp/tpchdata/
