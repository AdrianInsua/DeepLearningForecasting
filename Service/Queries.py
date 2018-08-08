query = {

    'prevision_ko': 'SELECT fecha_pedido, codigo_pais, codigo_almacen, id_tipo_envio, fecha_expedicion, limite_entrega_wcs, fecha_entrega > limite_entrega_wcs FROM entregas.pedidos_prevision_entrega WHERE $where',
    'prevision_pedidos': 'SELECT id_fecha, codigo_pais, sum(importe_neto) FROM drive.pedidos_tiempo WHERE codigo_pais=11 GROUP BY id_fecha, codigo_pais',
    'prevision_presupuesto': """
        select
            ds,
            y
        from
            drive.forecast_data s
    """,
    'paises': 'SELECT a.alpha3, a.alpha2 from maestros.pais p inner join maestros.all a on p.codigo_iso = a.alpha2'
}
