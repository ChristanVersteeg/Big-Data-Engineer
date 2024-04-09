DELIMITER $$

CREATE PROCEDURE `GetReviews`(
    IN `only_kaggle` BOOLEAN
)
BEGIN
    IF only_kaggle THEN
        SELECT * FROM all_reviews WHERE `<enum 'Type'>` = 'Type.KAGGLE';
    ELSE
        SELECT * FROM all_reviews;
    END IF;
END$$

DELIMITER ;
