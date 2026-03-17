from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://172.19.144.1:7687", auth=("neo4j", "12345678"))
with driver.session(database="final-chevrolet") as s:
    r = s.run("RETURN 1 AS ok")
    print("연결:", r.single()["ok"])
    indexes = s.run("SHOW INDEXES").data()
    for idx in indexes:
        print(idx.get("name"), "|", idx.get("type"), "|", idx.get("state"))
driver.close()
