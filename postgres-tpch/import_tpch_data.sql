\c tpch

COPY supplier FROM '/tmp/tpchdata/supplier.tbl' WITH DELIMITER AS '|';
COPY part FROM '/tmp/tpchdata/part.tbl' WITH DELIMITER AS '|';
COPY partsupp FROM '/tmp/tpchdata/partsupp.tbl' WITH DELIMITER AS '|';
COPY customer FROM '/tmp/tpchdata/customer.tbl' WITH DELIMITER AS '|';
COPY orders FROM '/tmp/tpchdata/orders.tbl' WITH DELIMITER AS '|';
COPY lineitem FROM '/tmp/tpchdata/lineitem.tbl' WITH DELIMITER AS '|';
COPY nation FROM '/tmp/tpchdata/nation.tbl' WITH DELIMITER AS '|';
COPY region FROM '/tmp/tpchdata/region.tbl' WITH DELIMITER AS '|';
