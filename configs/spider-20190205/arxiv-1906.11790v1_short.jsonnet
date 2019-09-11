(import 'arxiv-1906.11790v1.jsonnet') {
    local PREFIX = "data/spider-20190205/",
    data+: {
        val: {
            name: 'spider',
            paths: [PREFIX + 'dev_short.json'],
            tables_paths: [PREFIX + 'tables.json'],
            db_path: PREFIX + 'database',
        },
    },
}