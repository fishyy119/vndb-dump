SELECT
    v.id AS vn_id,
    COALESCE(
        jsonb_agg(DISTINCT rp.pid ORDER BY rp.pid)
            FILTER (WHERE rp.pid IS NOT NULL),
        '[]'::jsonb
    ) AS developers
FROM vn AS v
LEFT JOIN releases_vn AS rv
    ON rv.vid = v.id
LEFT JOIN releases_producers AS rp
    ON rv.id = rp.id
    AND rp.developer = TRUE
GROUP BY v.id
ORDER BY v.id;