CREATE CONSTRAINT brand_name_unique IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE;
CREATE CONSTRAINT model_key_unique IF NOT EXISTS FOR (m:Model) REQUIRE m.key IS UNIQUE;
CREATE CONSTRAINT manual_id_unique IF NOT EXISTS FOR (m:Manual) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT section_id_unique IF NOT EXISTS FOR (s:Section) REQUIRE s.id IS UNIQUE;
CREATE CONSTRAINT page_id_unique IF NOT EXISTS FOR (p:Page) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT entity_key_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE;
CREATE CONSTRAINT symptom_key_unique IF NOT EXISTS FOR (s:Symptom) REQUIRE s.key IS UNIQUE;
CREATE CONSTRAINT action_key_unique IF NOT EXISTS FOR (a:Action) REQUIRE a.key IS UNIQUE;
CREATE CONSTRAINT dtc_key_unique IF NOT EXISTS FOR (d:DTC) REQUIRE d.code IS UNIQUE;

CREATE INDEX model_name_idx IF NOT EXISTS FOR (m:Model) ON (m.name);
CREATE INDEX manual_type_idx IF NOT EXISTS FOR (m:Manual) ON (m.manual_type);
CREATE INDEX page_no_idx IF NOT EXISTS FOR (p:Page) ON (p.page_no);
CREATE INDEX page_source_idx IF NOT EXISTS FOR (p:Page) ON (p.source_file, p.page_no);
CREATE INDEX chunk_source_idx IF NOT EXISTS FOR (c:Chunk) ON (c.source_file, c.page_no);
CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name);

CREATE VECTOR INDEX chunk_embedding_idx IF NOT EXISTS
FOR (c:Chunk)
ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: __EMBEDDING_DIM__,
    `vector.similarity_function`: 'cosine'
  }
};

CREATE FULLTEXT INDEX chunk_text_ft IF NOT EXISTS
FOR (c:Chunk)
ON EACH [c.text];
