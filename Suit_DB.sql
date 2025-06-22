-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Hôte : localhost
-- Généré le : sam. 14 juin 2025 à 12:12
-- Version du serveur : 10.4.28-MariaDB
-- Version de PHP : 8.1.17

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;



CREATE TABLE `facial_analyses` (
  `id` int(11) NOT NULL,
  `session_id` int(11) NOT NULL,
  `dominant_emotion` varchar(50) DEFAULT NULL,
  `emotion_timeline` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`emotion_timeline`)),
  `dominant_sentiment` varchar(50) DEFAULT NULL,
  `sentiment_timeline` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`sentiment_timeline`)),
  `duration_seconds` float DEFAULT NULL,
  `frames_analyzed` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Structure de la table `sessions`
--

CREATE TABLE `sessions` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `created_at` datetime DEFAULT current_timestamp(),
  `token` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Déchargement des données de la table `sessions`
--

INSERT INTO `sessions` (`id`, `user_id`, `created_at`, `token`) VALUES
(20, 5, '2025-06-13 11:19:33', 'fb4feb07-bc6c-4af5-bc98-b2a76cf8005b'),
(21, 5, '2025-06-13 12:05:09', '5a108db7-159e-4a3d-bc2b-a53b3ddb71ed'),
(22, 5, '2025-06-13 12:54:45', '553811d3-ecdf-4912-9b92-8c4be63df3ef'),
(23, 5, '2025-06-13 13:00:18', '806e8dd2-7465-411d-adc4-541f428838d6'),
(24, 5, '2025-06-13 13:59:24', 'e1e1ec62-c2bd-4377-8035-28f744c328c4'),
(25, 5, '2025-06-13 14:28:41', '5ed26909-3a3f-4559-944a-603c28032310'),
(26, 5, '2025-06-13 19:34:53', 'c2411b05-1b3d-4303-b376-bf20954fc8fc'),
(27, 5, '2025-06-14 08:02:25', 'a2f3575d-1d47-41d9-9b6b-c11add7f251f'),
(28, 5, '2025-06-14 08:58:30', '5c6f26a8-813f-469a-a8f7-ce6b919c285d');

-- --------------------------------------------------------

--
-- Structure de la table `text_sentiments`
--

CREATE TABLE `text_sentiments` (
  `id` int(11) NOT NULL,
  `session_id` int(11) NOT NULL,
  `sentiment_label` varchar(50) DEFAULT NULL,
  `confidence_score` float DEFAULT NULL,
  `raw_scores` longtext CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL CHECK (json_valid(`raw_scores`))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Structure de la table `transcriptions`
--

CREATE TABLE `transcriptions` (
  `id` int(11) NOT NULL,
  `session_id` int(11) NOT NULL,
  `text` text NOT NULL,
  `created_at` datetime DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

-- --------------------------------------------------------

--
-- Structure de la table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `first_name` varchar(100) NOT NULL,
  `last_name` varchar(100) NOT NULL,
  `birth_date` date NOT NULL,
  `education_level` varchar(100) DEFAULT NULL,
  `target_position` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Déchargement des données de la table `users`
--

INSERT INTO `users` (`id`, `first_name`, `last_name`, `birth_date`, `education_level`, `target_position`) VALUES
(5, 'Meriem', 'HAMDANE', '2001-09-01', 'Master', 'Data Analyste');

-- --------------------------------------------------------

--
-- Structure de la table `user_credentials`
--

CREATE TABLE `user_credentials` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `email` varchar(150) NOT NULL,
  `hashed_password` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Déchargement des données de la table `user_credentials`
--

INSERT INTO `user_credentials` (`id`, `user_id`, `email`, `hashed_password`) VALUES
(5, 5, 'meriemhmdn0901@gmail.com', '$2b$12$3AJPNUx3Qe7ApPRUilLy8uP9zMY1Uhf60ubuxWkwx4k7wUGksmEtq');

--
-- Index pour les tables déchargées
--

--
-- Index pour la table `facial_analyses`
--
ALTER TABLE `facial_analyses`
  ADD PRIMARY KEY (`id`),
  ADD KEY `session_id` (`session_id`);

--
-- Index pour la table `sessions`
--
ALTER TABLE `sessions`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `token` (`token`),
  ADD KEY `user_id` (`user_id`);

--
-- Index pour la table `text_sentiments`
--
ALTER TABLE `text_sentiments`
  ADD PRIMARY KEY (`id`),
  ADD KEY `session_id` (`session_id`);

--
-- Index pour la table `transcriptions`
--
ALTER TABLE `transcriptions`
  ADD PRIMARY KEY (`id`),
  ADD KEY `session_id` (`session_id`);

--
-- Index pour la table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`);

--
-- Index pour la table `user_credentials`
--
ALTER TABLE `user_credentials`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `email` (`email`),
  ADD KEY `user_id` (`user_id`);

--
-- AUTO_INCREMENT pour les tables déchargées
--

--
-- AUTO_INCREMENT pour la table `facial_analyses`
--
ALTER TABLE `facial_analyses`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT pour la table `sessions`
--
ALTER TABLE `sessions`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=29;

--
-- AUTO_INCREMENT pour la table `text_sentiments`
--
ALTER TABLE `text_sentiments`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT pour la table `transcriptions`
--
ALTER TABLE `transcriptions`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT;

--
-- AUTO_INCREMENT pour la table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT pour la table `user_credentials`
--
ALTER TABLE `user_credentials`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- Contraintes pour les tables déchargées
--

--
-- Contraintes pour la table `facial_analyses`
--
ALTER TABLE `facial_analyses`
  ADD CONSTRAINT `facial_analyses_ibfk_1` FOREIGN KEY (`session_id`) REFERENCES `sessions` (`id`) ON DELETE CASCADE;

--
-- Contraintes pour la table `sessions`
--
ALTER TABLE `sessions`
  ADD CONSTRAINT `sessions_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE;

--
-- Contraintes pour la table `text_sentiments`
--
ALTER TABLE `text_sentiments`
  ADD CONSTRAINT `text_sentiments_ibfk_1` FOREIGN KEY (`session_id`) REFERENCES `sessions` (`id`) ON DELETE CASCADE;

--
-- Contraintes pour la table `transcriptions`
--
ALTER TABLE `transcriptions`
  ADD CONSTRAINT `transcriptions_ibfk_1` FOREIGN KEY (`session_id`) REFERENCES `sessions` (`id`) ON DELETE CASCADE;

--
-- Contraintes pour la table `user_credentials`
--
ALTER TABLE `user_credentials`
  ADD CONSTRAINT `user_credentials_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
