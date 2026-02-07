def requete_nettoyage_post_eda():

    requete = """
    --sql

    with premiere_jointure as (

        select dh.*,
               snb.min_nbrooms,
               snb.q1_nbrooms,
               snb.mediane_nbrooms,
               snb.q3_nbrooms,
               snb.max_nbrooms
               

        from df dh left join stats_nbrooms snb on
             dh.property_type = snb.property_type
    ),

        deuxieme_jointure as (
        
            select j1.*,
                   ss.min_size,
                   ss.q1_size,
                   ss.mediane_size,
                   ss.q3_size,
                   ss.max_size

            from premiere_jointure j1 left join stats_size ss on 
                 j1.property_type = ss.property_type
        
        ),

            gestion_nb_rooms as (
            
                select *,
                       case
                            when property_type = 'appartement' then
                                case
                                    when nb_rooms < 1 then 1
                                    when nb_rooms > 7 then 7
                                                      else nb_rooms
                                end

                            when property_type = 'ferme' then
                                case
                                    when nb_rooms < 3 then 3
                                    when nb_rooms > 16 then 16
                                                      else nb_rooms
                                end

                            when property_type = 'loft' then
                                case
                                    when nb_rooms < 2 then 2
                                                      else nb_rooms
                                end

                            when property_type = 'maison' then
                                case
                                    when nb_rooms < 2 then 2
                                    when nb_rooms > 13 then 13
                                                      else nb_rooms
                                end

                            when property_type = 'propriété' then
                                case
                                    when nb_rooms < 2 then 2
                                    when nb_rooms > 16 then 16
                                                      else nb_rooms
                                end

                            when property_type = 'viager' then
                                case
                                    when nb_rooms < 1 then 1
                                    when nb_rooms > 7 then 7
                                                      else nb_rooms
                                end

                            when property_type = 'villa' then
                                case
                                    when nb_rooms < 3 then 3
                                    when nb_rooms > 16 then 16
                                                      else nb_rooms
                                end
                            else nb_rooms
                        end as nb_rooms_corrige

                            
                from deuxieme_jointure

                where property_type not in ('divers','parking','terrain','terrain à bâtir')

            ),

                gestion_size as (
                
                    select *,
                       case
                            when property_type = 'appartement' then
                                case 
                                    when size > 200 then
                                        case
                                            when nb_rooms_corrige > q3_nbrooms then 200
                                                                               else q3_size
                                        end
                                    when size < 10 then
                                        case
                                            when nb_rooms_corrige < q1_nbrooms then 10
                                                                               else q1_size
                                        end
                                    else size
                                end

                            when property_type = 'chalet' then
                                case 
                                    when size > 300 then
                                        case
                                            when nb_rooms_corrige > q3_nbrooms then 300
                                                                               else q3_size
                                        end
                                    when size < 30 then
                                        case
                                            when nb_rooms_corrige < q1_nbrooms then 30
                                                                               else q1_size
                                        end
                                    else size
                                end

                            when property_type = 'chambre' then 13

                            when property_type = 'duplex' then
                                case 
                                    when size > 210 then
                                        case
                                            when nb_rooms_corrige > q3_nbrooms then 210
                                                                               else q3_size
                                        end
                                    when size < 25 then
                                        case
                                            when nb_rooms_corrige < q1_nbrooms then 25
                                                                               else q1_size
                                        end
                                    else size
                                end

                            when property_type = 'ferme' then
                                case 
                                    when size > 400 then
                                        case
                                            when nb_rooms_corrige > q3_nbrooms then 400
                                                                               else q3_size
                                        end
                                    when size < 100 then
                                        case
                                            when nb_rooms_corrige < q1_nbrooms then 100
                                                                               else q1_size
                                        end
                                    else size
                                end

                            when property_type = 'loft' then
                                case 
                                    when size > 350 then
                                        case
                                            when nb_rooms_corrige > q3_nbrooms then 350
                                                                               else q3_size
                                        end
                                    when size < 36 then
                                        case
                                            when nb_rooms_corrige < q1_nbrooms then 36
                                                                               else q1_size
                                        end
                                    else size
                                end

                            when property_type = 'maison' then
                                case 
                                    when size > 300 then
                                        case
                                            when nb_rooms_corrige > q3_nbrooms then 300
                                                                               else q3_size
                                        end
                                    when size < 30 then
                                        case
                                            when nb_rooms_corrige < q1_nbrooms then 30
                                                                               else q1_size
                                        end
                                    else size
                                end

                            when property_type = 'manoir' then
                                case 
                                    when size > 600 then
                                        case
                                            when nb_rooms_corrige > q3_nbrooms then 600
                                                                               else q3_size
                                        end
                                    when size < 200 then
                                        case
                                            when nb_rooms_corrige < q1_nbrooms then 200
                                                                               else q1_size
                                        end
                                    else size
                                end

                            when property_type = 'moulin' then
                                case 
                                    when size > 400 then
                                        case
                                            when nb_rooms_corrige > q3_nbrooms then 400
                                                                               else q3_size
                                        end
                                    when size < 80 then
                                        case
                                            when nb_rooms_corrige < q1_nbrooms then 80
                                                                               else q1_size
                                        end
                                    else size
                                end

                            when property_type = 'propriété' then
                                case 
                                    when size > 600 then
                                        case
                                            when nb_rooms_corrige > q3_nbrooms then 600
                                                                               else q3_size
                                        end
                                    when size < 150 then
                                        case
                                            when nb_rooms_corrige < q1_nbrooms then 150
                                                                               else q1_size
                                        end
                                    else size
                                end

                            when property_type = 'villa' then
                                case 
                                    when size > 300 then
                                        case
                                            when nb_rooms_corrige > q3_nbrooms then 300
                                                                               else q3_size
                                        end
                                    when size < 80 then
                                        case
                                            when nb_rooms_corrige < q1_nbrooms then 80
                                                                               else q1_size
                                        end
                                    else size
                                end
                            else size
                        end as size_corrige

                    from gestion_nb_rooms
                
                )
                    select *
                    from gestion_size"""
    

    return requete