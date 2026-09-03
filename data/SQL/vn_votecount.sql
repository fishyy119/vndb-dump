SELECT
    v.id AS vid,
    v.c_votecount AS votecount,
    RANK() OVER (ORDER BY v.c_votecount DESC) AS votecount_rank
FROM vn AS v
ORDER BY votecount_rank;